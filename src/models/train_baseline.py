"""Treino e avaliacao do modelo baseline de atraso por pedido."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.data.ingest import ingest
from src.features.build_features import build_features


FEATURE_COLUMNS: tuple[str, ...] = (
    "qtd_itens", "peso_total_g", "categoria_principal",
    "qtd_categorias_distintas", "distancia_media_km", "distancia_max_km",
    "estado_cliente", "estado_vendedor_principal", "estado_vendedor_mais_distante",
    "qtd_estados_vendedor", "forma_pagamento", "num_parcelas", "horas_aprovacao",
    "dia_semana_compra", "mes_compra", "dia_do_mes", "quinzena_compra",
    "prazo_estimado_dias",
)

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "categoria_principal", "estado_cliente", "estado_vendedor_principal",
    "estado_vendedor_mais_distante", "forma_pagamento", "dia_semana_compra",
    "mes_compra", "quinzena_compra",
)


@dataclass(frozen=True)
class TrainingArtifacts:
    """Artefatos do treino necessarios para analises posteriores."""

    metrics: dict[str, dict[str, float | dict[str, float]]]
    model: HistGradientBoostingClassifier
    x_test: pd.DataFrame


def temporal_split(
    data: pd.DataFrame,
    proportions: Sequence[float] = (0.7, 0.15, 0.15),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa os dados em treino, validacao e teste na ordem da compra."""
    if len(proportions) != 3 or any(proportion <= 0 for proportion in proportions):
        raise ValueError("proportions deve conter tres valores positivos")
    if not abs(sum(proportions) - 1.0) < 1e-9:
        raise ValueError("as proporcoes devem somar 1")
    if "order_purchase_timestamp" not in data:
        raise KeyError("order_purchase_timestamp nao encontrado")
    ordered = data.sort_values("order_purchase_timestamp", kind="mergesort")
    train_end = int(len(ordered) * proportions[0])
    validation_end = train_end + int(len(ordered) * proportions[1])
    if train_end == 0 or validation_end == train_end or validation_end == len(ordered):
        raise ValueError("as proporcoes precisam produzir tres conjuntos nao vazios")
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )


def _apply_shared_categories(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Aplica a uniao das categorias aos tres conjuntos, in-place."""
    for column in CATEGORICAL_COLUMNS:
        categories = pd.concat(
            [train[column], validation[column], test[column]], ignore_index=True
        ).dropna().unique()
        dtype = pd.api.types.CategoricalDtype(categories=categories)
        for frame in (train, validation, test):
            frame[column] = frame[column].astype(dtype)


def prepare_model_data(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Seleciona as features e padroniza categorias entre os splits."""
    frames = [train.copy(), validation.copy(), test.copy()]
    _apply_shared_categories(*frames)
    features = [frame.loc[:, FEATURE_COLUMNS] for frame in frames]
    targets = [frame["atrasou"].astype(int) for frame in frames]
    return (*features, *targets)


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    **classifier_parameters: object,
) -> HistGradientBoostingClassifier:
    """Treina o baseline usando validacao temporal para early stopping."""
    parameters = {
        "categorical_features": "from_dtype",
        "class_weight": "balanced",
        "early_stopping": True,
        "random_state": 42,
        **classifier_parameters,
    }
    model = HistGradientBoostingClassifier(**parameters)
    model.fit(x_train, y_train, X_val=x_validation, y_val=y_validation)
    return model


def evaluate_model(
    model: HistGradientBoostingClassifier,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float | dict[str, float]]:
    """Calcula metricas e a distribuicao do alvo."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    class_proportions = target.value_counts(normalize=True).reindex([0, 1], fill_value=0)
    return {
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "f1_score": float(f1_score(target, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "class_proportions": {
            "nao_atrasou": float(class_proportions[0]),
            "atrasou": float(class_proportions[1]),
        },
    }


def load_features(raw_dir: str | Path, database_path: str | Path = ":memory:") -> pd.DataFrame:
    """Ingeste os CSVs, constroi features e retorna a tabela materializada."""
    connection = ingest(raw_dir, database_path)
    try:
        build_features(connection)
        return connection.sql("SELECT * FROM features").df()
    finally:
        connection.close()


def print_evaluation(metrics: dict[str, dict[str, float | dict[str, float]]]) -> None:
    """Imprime as metricas de cada particao de forma distinta."""
    for split_name in ("validation", "test"):
        split_metrics = metrics[split_name]
        proportions = split_metrics["class_proportions"]
        assert isinstance(proportions, dict)
        print(f"\nAvaliacao - {split_name}")
        print(f"recall: {split_metrics['recall']:.4f}")
        print(f"precision: {split_metrics['precision']:.4f}")
        print(f"f1_score: {split_metrics['f1_score']:.4f}")
        print(f"average_precision (AUC-PR): {split_metrics['average_precision']:.4f}")
        print(
            "proporcao de classes - "
            f"nao atrasou: {proportions['nao_atrasou']:.4f}, "
            f"atrasou: {proportions['atrasou']:.4f}"
        )


def run_training(
    raw_dir: str | Path,
    database_path: str | Path = ":memory:",
    return_artifacts: bool = False,
) -> dict[str, dict[str, float | dict[str, float]]] | TrainingArtifacts:
    """Executa o pipeline completo e retorna metricas de validacao e teste."""
    data = load_features(raw_dir, database_path)
    train, validation, test = temporal_split(data)
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_model_data(
        train, validation, test
    )
    model = train_model(x_train, y_train, x_validation, y_validation)
    metrics = {
        "validation": evaluate_model(model, x_validation, y_validation),
        "test": evaluate_model(model, x_test, y_test),
    }
    print_evaluation(metrics)
    if return_artifacts:
        return TrainingArtifacts(metrics=metrics, model=model, x_test=x_test)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--database-path", type=Path, default=Path(":memory:"))
    args = parser.parse_args()
    run_training(args.raw_dir, args.database_path)


if __name__ == "__main__":
    main()