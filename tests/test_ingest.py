from pathlib import Path

from src.data.ingest import RAW_FILES, ingest


def test_orders_labeled_has_binary_target_and_non_null_dates(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    orders = (
        "order_id,order_status,order_delivered_customer_date,"
        "order_estimated_delivery_date\n"
        "1,delivered,2017-01-11,2017-01-10\n"
        "2,delivered,2017-01-10,2017-01-10\n"
        "3,delivered,,2017-01-10\n"
        "4,shipped,2017-01-11,2017-01-10\n"
    )
    for view_name, file_name in RAW_FILES.items():
        content = orders if view_name == "orders" else "id\n1\n"
        (raw_dir / file_name).write_text(content, encoding="utf-8")

    connection = ingest(raw_dir)
    try:
        invalid_targets = connection.sql(
            """
            SELECT COUNT(*)
            FROM orders_labeled
            WHERE atrasou NOT IN (0, 1) OR atrasou IS NULL
            """
        ).fetchone()[0]
        null_dates = connection.sql(
            """
            SELECT COUNT(*)
            FROM orders_labeled
            WHERE order_delivered_customer_date IS NULL
               OR order_estimated_delivery_date IS NULL
            """
        ).fetchone()[0]
        non_delivered_orders = connection.sql(
            """
            SELECT COUNT(*)
            FROM orders_labeled
            WHERE order_status <> 'delivered'
            """
        ).fetchone()[0]

        assert invalid_targets == 0
        assert null_dates == 0
        assert non_delivered_orders == 0
    finally:
        connection.close()