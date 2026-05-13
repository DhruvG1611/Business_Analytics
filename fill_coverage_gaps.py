import sys
import os
import yaml

import config
from core.metric_generator import auto_generate_metric, SAKILA_TABLES

# Initialize config
config.csm = yaml.safe_load(open('csm_enterprise.yaml', 'r', encoding='utf-8'))
config.glossary = yaml.safe_load(open('bgo.yaml', 'r', encoding='utf-8'))
config.METRIC_KEY_MAP = {k.lower(): k for k in config.csm.get('metrics', {}).keys()}

def generate_variants(base_metric: str, base_table: str, dimension_key: str, measure_expr: str):
    """Generates SUM, COUNT, and AVG variants if applicable."""
    
    # 1. SUM Variant
    auto_generate_metric(
        metric_key=base_metric,
        base_table=base_table,
        dimension_key=dimension_key,
        measure_expr=measure_expr,
        compute_func="SUM",
        label=base_metric.replace('_', ' ').title()
    )
    
    # 2. COUNT Variant
    count_metric = base_metric.replace('revenue', 'rentals') if 'revenue' in base_metric else f"{base_metric}_count"
    count_measure = f"{base_table}.{base_table}_id"
    if count_metric not in config.csm['metrics']:
        auto_generate_metric(
            metric_key=count_metric,
            base_table=base_table,
            dimension_key=dimension_key,
            measure_expr=count_measure,
            compute_func="COUNT",
            label=count_metric.replace('_', ' ').title()
        )
        
    # 3. AVG Variant
    avg_metric = base_metric.replace('revenue', 'average_revenue') if 'revenue' in base_metric else f"average_{base_metric}"
    if avg_metric not in config.csm['metrics']:
        auto_generate_metric(
            metric_key=avg_metric,
            base_table=base_table,
            dimension_key=dimension_key,
            measure_expr=measure_expr,
            compute_func="AVG",
            label=avg_metric.replace('_', ' ').title()
        )

def fill_gaps():
    metrics_to_fill = [
        # REVENUE METRICS
        ("revenue_by_category", "payment", "category", "payment.amount"),
        ("revenue_by_film", "payment", "film", "payment.amount"),
        ("revenue_by_store", "payment", "store", "payment.amount"),
        ("revenue_by_customer", "payment", "customer", "payment.amount"),
        ("revenue_by_staff", "payment", "staff", "payment.amount"),
        
        # COUNT METRICS 
        ("rentals_by_category", "rental", "category", "rental.rental_id"),
        ("rentals_by_film", "rental", "film", "rental.rental_id"),
        ("rentals_by_store", "rental", "store", "rental.rental_id"),
        ("rentals_by_customer", "rental", "customer", "rental.rental_id"),
        ("rentals_by_staff", "rental", "staff", "rental.rental_id"),
        
        # INVENTORY METRICS
        ("film_count_by_category", "film_category", "category", "film_category.film_id"),
        ("inventory_count_by_store", "inventory", "store", "inventory.inventory_id"),
        
        # CUSTOMER METRICS
        ("customer_count_by_city", "customer", "city", "customer.customer_id")
    ]
    
    for metric_key, base_table, dim_key, expr in metrics_to_fill:
        # Overwrite to ensure high-quality patterns and deterministic join paths
        print(f"Generating metric: {metric_key}")
        compute_func = "SUM" if "revenue" in metric_key else "COUNT"
        if "revenue" in metric_key:
            generate_variants(metric_key, base_table, dim_key, expr)
        else:
            auto_generate_metric(metric_key, base_table, dim_key, expr, compute_func=compute_func)

if __name__ == '__main__':
    fill_gaps()
