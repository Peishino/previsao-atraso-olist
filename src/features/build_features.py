"""Construcao das features de pedidos a partir das views DuckDB."""

import duckdb


def create_item_features(connection: duckdb.DuckDBPyConnection) -> None:
    """Agrega quantidade, peso, categorias e vendedores por pedido."""
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW item_features AS
        WITH item_details AS (
            SELECT
                items.order_id,
                items.price,
                COALESCE(
                    category_translation.product_category_name_english,
                    products.product_category_name,
                    'unknown'
                ) AS categoria,
                products.product_weight_g AS peso_g,
                sellers.seller_state AS estado_vendedor
            FROM order_items AS items
            LEFT JOIN products
                ON products.product_id = items.product_id
            LEFT JOIN category_translation
                ON category_translation.product_category_name =
                   products.product_category_name
            LEFT JOIN sellers
                ON sellers.seller_id = items.seller_id
        )
        SELECT
            order_id,
            COUNT(*) AS qtd_itens,
            SUM(peso_g) AS peso_total_g,
            STRING_AGG(DISTINCT categoria, ', ') AS categorias_lista,
            arg_max(categoria, price) AS categoria_principal,
            COUNT(DISTINCT categoria) AS qtd_categorias_distintas,
            STRING_AGG(DISTINCT estado_vendedor, ', ') AS estados_vendedor_lista,
            arg_max(estado_vendedor, price) AS estado_vendedor_principal,
            COUNT(DISTINCT estado_vendedor) AS qtd_estados_vendedor
        FROM item_details
        GROUP BY order_id
        """
    )


def create_payment_features(connection: duckdb.DuckDBPyConnection) -> None:
    """Agrega formas de pagamento e parcelas por pedido."""
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW payment_features AS
        SELECT
            order_id,
            STRING_AGG(DISTINCT payment_type, ', ') AS forma_pagamento,
            MAX(payment_installments) AS num_parcelas
        FROM payments
        GROUP BY order_id
        """
    )


def create_location_features(connection: duckdb.DuckDBPyConnection) -> None:
    """Calcula distancia media e estados por pedido."""
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW geolocation_by_zip AS
        SELECT
            geolocation_zip_code_prefix,
            AVG(geolocation_lat) AS latitude,
            AVG(geolocation_lng) AS longitude
        FROM geolocation
        GROUP BY geolocation_zip_code_prefix
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW location_features AS
        WITH item_locations AS (
            SELECT
                items.order_id,
                customers.customer_state AS estado_cliente,
                sellers.seller_state AS estado_vendedor,
                customer_geo.latitude AS customer_latitude,
                customer_geo.longitude AS customer_longitude,
                seller_geo.latitude AS seller_latitude,
                seller_geo.longitude AS seller_longitude
            FROM order_items AS items
            INNER JOIN orders_labeled AS orders
                ON orders.order_id = items.order_id
            INNER JOIN customers
                ON customers.customer_id = orders.customer_id
            LEFT JOIN sellers
                ON sellers.seller_id = items.seller_id
            LEFT JOIN geolocation_by_zip AS customer_geo
                ON customer_geo.geolocation_zip_code_prefix =
                   customers.customer_zip_code_prefix
            LEFT JOIN geolocation_by_zip AS seller_geo
                ON seller_geo.geolocation_zip_code_prefix =
                   sellers.seller_zip_code_prefix
        ), item_distances AS (
            SELECT
                order_id,
                estado_cliente,
                estado_vendedor,
                2 * 6371 * ASIN(SQRT(
                    POWER(SIN(RADIANS(seller_latitude - customer_latitude) / 2), 2)
                    + COS(RADIANS(customer_latitude))
                    * COS(RADIANS(seller_latitude))
                    * POWER(SIN(RADIANS(seller_longitude - customer_longitude) / 2), 2)
                )) AS distancia_km
            FROM item_locations
        )
        SELECT
            order_id,
            MAX(estado_cliente) AS estado_cliente,
            STRING_AGG(DISTINCT estado_vendedor, ', ') AS estados_vendedor_lista,
            AVG(distancia_km) AS distancia_media_km,
            arg_max(estado_vendedor, distancia_km) AS estado_vendedor_mais_distante,
            MAX(distancia_km) AS distancia_max_km
        FROM item_distances
        GROUP BY order_id
        """
    )


def create_time_features(connection: duckdb.DuckDBPyConnection) -> None:
    """Extrai tempo de aprovacao, calendario e prazo estimado da compra."""
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW time_features AS
        SELECT
            order_id,
            DATE_DIFF(
                'minute', order_purchase_timestamp, order_approved_at
            ) / 60.0 AS horas_aprovacao,
            DAYOFWEEK(order_purchase_timestamp) AS dia_semana_compra,
            MONTH(order_purchase_timestamp) AS mes_compra,
            DAY(order_purchase_timestamp) AS dia_do_mes,
            CASE
                WHEN DAY(order_purchase_timestamp) <= 15 THEN 1
                ELSE 2
            END AS quinzena_compra,
            DATE_DIFF(
                'day', order_purchase_timestamp, order_estimated_delivery_date
            ) AS prazo_estimado_dias
        FROM orders_labeled
        """
    )


def build_features(connection: duckdb.DuckDBPyConnection) -> None:
    """Cria features sem perder pedidos rotulados.

    As colunas order_delivered_customer_date e order_status, vindas de
    orders.*, foram usadas para criar o alvo ``atrasou`` e nao podem ser
    usadas como features do modelo. A coluna
    order_estimated_delivery_date so pode ser usada indiretamente por meio de
    ``prazo_estimado_dias``.
    """
    create_item_features(connection)
    create_payment_features(connection)
    create_location_features(connection)
    create_time_features(connection)
    connection.execute(
        """
        CREATE OR REPLACE TABLE features AS
        SELECT
            orders.*,
            item_features.qtd_itens,
            item_features.peso_total_g,
            item_features.categorias_lista,
            item_features.categoria_principal,
            item_features.qtd_categorias_distintas,
            payment_features.forma_pagamento,
            payment_features.num_parcelas,
            location_features.distancia_media_km,
            location_features.estado_cliente,
            COALESCE(
                location_features.estados_vendedor_lista,
                item_features.estados_vendedor_lista
            ) AS estados_vendedor_lista,
            item_features.estado_vendedor_principal,
            item_features.qtd_estados_vendedor,
            location_features.estado_vendedor_mais_distante,
            location_features.distancia_max_km,
            time_features.horas_aprovacao,
            time_features.dia_semana_compra,
            time_features.mes_compra,
            time_features.dia_do_mes,
            time_features.quinzena_compra,
            time_features.prazo_estimado_dias
        FROM orders_labeled AS orders
        LEFT JOIN item_features USING (order_id)
        LEFT JOIN payment_features USING (order_id)
        LEFT JOIN location_features USING (order_id)
        LEFT JOIN time_features USING (order_id)
        """
    )