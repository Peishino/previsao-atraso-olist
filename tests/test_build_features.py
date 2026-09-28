import csv
from numbers import Real
from pathlib import Path
from typing import Sequence

import pytest

from src.data.ingest import RAW_FILES, ingest
from src.features.build_features import build_features


EXPECTED_COLUMNS = {
    "distancia_media_km",
    "horas_aprovacao",
    "qtd_itens",
    "peso_total_g",
    "categorias_lista",
    "categoria_principal",
    "qtd_categorias_distintas",
    "forma_pagamento",
    "num_parcelas",
    "estado_cliente",
    "estados_vendedor_lista",
    "estado_vendedor_principal",
    "qtd_estados_vendedor",
    "estado_vendedor_mais_distante",
    "distancia_max_km",
    "dia_semana_compra",
    "mes_compra",
    "dia_do_mes",
    "quinzena_compra",
    "prazo_estimado_dias",
}


def _write_csv(
    raw_dir: Path,
    file_name: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    with (raw_dir / file_name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def _create_fake_raw_data(raw_dir: Path) -> None:
    files = {
        "olist_orders_dataset.csv": (
            (
                "order_id",
                "order_status",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "customer_id",
                "order_purchase_timestamp",
                "order_approved_at",
            ),
            (
                ("A", "delivered", "2017-01-05 10:00:00", "2017-01-10", "customer_a", "2017-01-01 10:00:00", "2017-01-01 12:30:00"),
                ("B", "delivered", "2017-01-08 10:00:00", "2017-01-12", "customer_b", "2017-01-02 08:00:00", "2017-01-02 09:00:00"),
                ("C", "delivered", "2017-01-09 10:00:00", "2017-01-15", "customer_c", "2017-01-03 10:00:00", "2017-01-03 11:00:00"),
            ),
        ),
        "olist_customers_dataset.csv": (
            ("customer_id", "customer_zip_code_prefix", "customer_state"),
            (("customer_a", 10000, "SP"), ("customer_b", 30000, "MG"), ("customer_c", 60000, "BA")),
        ),
        "olist_order_items_dataset.csv": (
            ("order_id", "order_item_id", "product_id", "seller_id", "price"),
            (("A", 1, "product_a", "seller_a", 10), ("B", 1, "product_b1", "seller_b1", 15), ("B", 2, "product_b2", "seller_b2", 30), ("C", 1, "product_c", "seller_c", 20)),
        ),
        "olist_order_payments_dataset.csv": (
            ("order_id", "payment_type", "payment_installments"),
            (("A", "credit_card", 1), ("B", "boleto", 2), ("C", "credit_card", 1)),
        ),
        "olist_products_dataset.csv": (
            ("product_id", "product_category_name", "product_weight_g"),
            (("product_a", "cat_a", 100), ("product_b1", "cat_b", 200), ("product_b2", "cat_c", 300), ("product_c", "cat_a", 150)),
        ),
        "olist_sellers_dataset.csv": (
            ("seller_id", "seller_state", "seller_zip_code_prefix"),
            (("seller_a", "SP", 20000), ("seller_b1", "RJ", 40000), ("seller_b2", "ES", 50000), ("seller_c", "BA", 70000)),
        ),
        "olist_geolocation_dataset.csv": (
            ("geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"),
            ((10000, -23.5, -46.6), (20000, -23.6, -46.7), (30000, -19.9, -44.0), (40000, -22.9, -43.2), (50000, -20.3, -40.3), (60000, -12.9, -38.5)),
        ),
        "product_category_name_translation.csv": (
            ("product_category_name", "product_category_name_english"),
            (("cat_a", "category_a"), ("cat_b", "category_b"), ("cat_c", "category_c")),
        ),
    }
    for file_name, (columns, rows) in files.items():
        _write_csv(raw_dir, file_name, columns, rows)
    _write_csv(raw_dir, RAW_FILES["reviews"], ("review_id",), (("review_a",),))


def test_features_build_from_minimal_raw_data(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _create_fake_raw_data(raw_dir)

    connection = ingest(raw_dir)
    try:
        build_features(connection)
        orders_count = connection.sql(
            "SELECT COUNT(*) FROM orders_labeled"
        ).fetchone()[0]
        features_count = connection.sql(
            "SELECT COUNT(*) FROM features"
        ).fetchone()[0]
        feature_columns = {
            row[0] for row in connection.sql("DESCRIBE features").fetchall()
        }

        assert features_count == orders_count
        assert EXPECTED_COLUMNS <= feature_columns

        order_b = connection.sql(
            """
            SELECT qtd_estados_vendedor, qtd_categorias_distintas, qtd_itens,
                   categoria_principal, estado_vendedor_principal,
                   distancia_max_km, distancia_media_km
            FROM features
            WHERE order_id = 'B'
            """
        ).fetchone()
        assert order_b[:3] == (2, 2, 2)
        assert order_b[3:5] == ("category_c", "ES")
        assert order_b[5] >= order_b[6]

        order_a = connection.sql(
            """
            SELECT qtd_estados_vendedor, distancia_max_km, distancia_media_km
            FROM features
            WHERE order_id = 'A'
            """
        ).fetchone()
        assert order_a[0] == 1
        assert order_a[1] == pytest.approx(order_a[2])

        order_c = connection.sql(
            "SELECT distancia_max_km FROM features WHERE order_id = 'C'"
        ).fetchone()
        assert order_c[0] is None

        time_values = connection.sql(
            """
            SELECT horas_aprovacao, quinzena_compra, prazo_estimado_dias
            FROM features
            """
        ).fetchall()
        assert all(isinstance(row[0], Real) and row[0] >= 0 for row in time_values)
        assert {row[1] for row in time_values} <= {1, 2}
        assert all(row[2] > 0 for row in time_values)
    finally:
        connection.close()