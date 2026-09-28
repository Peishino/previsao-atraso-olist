"""Ingestao dos dados brutos de pedidos no DuckDB."""

from pathlib import Path

import duckdb


RAW_FILES: dict[str, str] = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def connect_database(database_path: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    """Abre uma conexao DuckDB em memoria ou no arquivo informado."""
    return duckdb.connect(str(database_path))


def _sql_path(file_path: Path) -> str:
    """Converte um caminho local para uma string segura dentro de uma query SQL."""
    return str(file_path.resolve()).replace("'", "''")


def create_raw_views(
    connection: duckdb.DuckDBPyConnection,
    raw_dir: Path,
) -> None:
    """Cria uma view DuckDB para cada CSV bruto esperado."""
    for view_name, file_name in RAW_FILES.items():
        file_path = raw_dir / file_name
        connection.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT * FROM read_csv_auto('{_sql_path(file_path)}')"
        )


def create_labeled_orders(connection: duckdb.DuckDBPyConnection) -> None:
    """Materializa pedidos entregues com o alvo binario de atraso."""
    connection.execute(
        """
        CREATE OR REPLACE TABLE orders_labeled AS
        SELECT
            *,
            CASE
                WHEN order_delivered_customer_date > order_estimated_delivery_date
                THEN 1
                ELSE 0
            END AS atrasou
        FROM orders
        WHERE order_status = 'delivered'
          AND order_delivered_customer_date IS NOT NULL
          AND order_estimated_delivery_date IS NOT NULL
        """
    )


def ingest(
    raw_dir: str | Path,
    database_path: str | Path = ":memory:",
) -> duckdb.DuckDBPyConnection:
    """Cria as views brutas e a tabela orders_labeled.

    A conexao e devolvida aberta para que as etapas seguintes possam reutilizar
    as views e a tabela na mesma sessao DuckDB.
    """
    connection = connect_database(database_path)
    create_raw_views(connection, Path(raw_dir))
    create_labeled_orders(connection)
    return connection