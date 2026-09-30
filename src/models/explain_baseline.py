"""Explicabilidade SHAP do modelo baseline de atraso por pedido."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.train_baseline import (
    CATEGORICAL_COLUMNS,
    TrainingArtifacts,
    run_training,
)


DEFAULT_OUTPUT_PATH = Path("reports/shap_summary.png")
TOP_FEATURE_COUNT = 5


def prepare_explanation_features(features: pd.DataFrame) -> pd.DataFrame:
    """Converte categorias para codigos numericos aceitos pelo TreeExplainer."""
    encoded = features.copy()
    for column in CATEGORICAL_COLUMNS:
        encoded[column] = encoded[column].cat.codes
    return encoded.astype(float)


def prepare_display_features(features: pd.DataFrame) -> pd.DataFrame:
    """Preserva os valores originais das categorias para o grafico."""
    display_features = features.copy()
    for column in CATEGORICAL_COLUMNS:
        display_features[column] = display_features[column].astype(str)
    return display_features


def calculate_shap_values(artifacts: TrainingArtifacts) -> np.ndarray:
    """Calcula os valores SHAP do conjunto de teste com TreeExplainer."""
    explainer = shap.TreeExplainer(artifacts.model)
    # O TreeExplainer exige numeros para calcular; o grafico pode exibir os valores originais.
    values: Any = explainer.shap_values(
        prepare_explanation_features(artifacts.x_test)
    )
    if isinstance(values, list):
        values = values[1]
    return np.asarray(values)


def save_summary_plot(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Salva o summary plot com a importancia e a direcao das features."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, features, show=False)
    plt.savefig(destination, bbox_inches="tight")
    plt.close()


def top_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    count: int = TOP_FEATURE_COUNT,
) -> list[tuple[str, float]]:
    """Retorna as features ordenadas pela media do SHAP absoluto."""
    mean_absolute_values = np.abs(shap_values).mean(axis=0)
    ranking = sorted(
        zip(feature_names, mean_absolute_values),
        key=lambda item: item[1],
        reverse=True,
    )
    return [(name, float(value)) for name, value in ranking[:count]]


def print_top_features(shap_values: np.ndarray, features: pd.DataFrame) -> None:
    """Imprime as cinco features mais importantes."""
    print("\nTop 5 features por media do valor absoluto de SHAP")
    for name, importance in top_features(shap_values, features.columns.tolist()):
        print(f"{name}: {importance:.6f}")


def explain_baseline(
    raw_dir: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    database_path: str | Path = ":memory:",
) -> None:
    """Treina o baseline, salva sua explicacao e imprime o ranking de features."""
    artifacts = run_training(raw_dir, database_path, return_artifacts=True)
    assert isinstance(artifacts, TrainingArtifacts)
    shap_values = calculate_shap_values(artifacts)
    display_features = prepare_display_features(artifacts.x_test)
    save_summary_plot(shap_values, display_features, output_path)
    print_top_features(shap_values, artifacts.x_test)
    print(f"\nGrafico salvo em: {Path(output_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--database-path", type=Path, default=Path(":memory:"))
    args = parser.parse_args()
    explain_baseline(args.raw_dir, args.output_path, args.database_path)


if __name__ == "__main__":
    main()