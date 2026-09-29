from datetime import datetime, timedelta

import pandas as pd

from src.models.train_baseline import (
    FEATURE_COLUMNS,
    evaluate_model,
    prepare_model_data,
    temporal_split,
    train_model,
)


def _fake_features(rows: int = 20) -> pd.DataFrame:
    timestamps = [datetime(2020, 1, 1) + timedelta(days=index) for index in range(rows)]
    data: dict[str, object] = {
        "order_purchase_timestamp": timestamps,
        "atrasou": [index % 2 for index in range(rows)],
    }
    for column in FEATURE_COLUMNS:
        data[column] = [f"value_{index % 3}" for index in range(rows)]
    for column in (
        "qtd_itens",
        "peso_total_g",
        "qtd_categorias_distintas",
        "distancia_media_km",
        "distancia_max_km",
        "qtd_estados_vendedor",
        "num_parcelas",
        "horas_aprovacao",
        "dia_do_mes",
        "prazo_estimado_dias",
    ):
        data[column] = [index + 1 for index in range(rows)]
    return pd.DataFrame(data)


def test_temporal_split_preserves_order() -> None:
    train, validation, test = temporal_split(_fake_features())

    assert train["order_purchase_timestamp"].max() <= validation[
        "order_purchase_timestamp"
    ].min()
    assert validation["order_purchase_timestamp"].max() <= test[
        "order_purchase_timestamp"
    ].min()


def test_baseline_pipeline_returns_separate_metrics() -> None:
    train, validation, test = temporal_split(_fake_features())
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_model_data(
        train, validation, test
    )
    model = train_model(
        x_train,
        y_train,
        x_validation,
        y_validation,
        max_iter=10,
        min_samples_leaf=1,
    )

    metrics = {
        "validation": evaluate_model(model, x_validation, y_validation),
        "test": evaluate_model(model, x_test, y_test),
    }

    assert set(metrics) == {"validation", "test"}
    assert {
        "recall",
        "precision",
        "f1_score",
        "average_precision",
        "class_proportions",
    } <= set(metrics["validation"])
    assert x_train.columns.tolist() == list(FEATURE_COLUMNS)