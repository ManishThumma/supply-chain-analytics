from __future__ import annotations

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

PII_COLUMNS = [
    "Customer Email",
    "Customer Password",
    "Customer Fname",
    "Customer Lname",
]

DATE_COLUMNS = {
    "order date (DateOrders)": "order_date",
    "shipping date (DateOrders)": "ship_date",
}

COLUMN_RENAMES = {
    "Type": "payment_type",
    "Days for shipping (real)": "days_shipping_real",
    "Days for shipment (scheduled)": "days_shipping_scheduled",
    "Delivery Status": "delivery_status",
    "Late_delivery_risk": "late_delivery_risk",
    "Category Name": "category_name",
    "Customer Segment": "customer_segment",
    "Customer City": "customer_city",
    "Customer State": "customer_state",
    "Customer Country": "customer_country",
    "Department Name": "department_name",
    "Market": "market",
    "Order Item Quantity": "order_quantity",
    "Order Item Total": "order_item_total",
    "Order Profit Per Order": "order_profit",
    "Sales": "sales",
    "Shipping Mode": "shipping_mode",
    "Order Status": "order_status",
}

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        parquet = _DATA_DIR / "dataco_supply_chain.parquet"
        csv = _DATA_DIR / "dataco_supply_chain.csv"
        path = parquet if parquet.exists() else csv

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download DataCo Smart Supply Chain from Kaggle and place it in /data."
        )

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        logger.info("Loaded %d rows from %s", len(df), path.name)
        return df

    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path)

    drop_cols = [c for c in PII_COLUMNS if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    for raw_col, new_name in DATE_COLUMNS.items():
        if raw_col in df.columns:
            df[new_name] = pd.to_datetime(df[raw_col], errors="coerce")
            df.drop(columns=[raw_col], inplace=True)

    rename_map = {k: v for k, v in COLUMN_RENAMES.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)

    null_counts = df.isnull().sum()
    high_null = null_counts[null_counts / len(df) > 0.5]
    if not high_null.empty:
        logger.warning("Dropping %d columns with >50%% nulls: %s", len(high_null), high_null.index.tolist())
        df.drop(columns=high_null.index, inplace=True)

    logger.info("Clean dataset: %d rows, %d columns", len(df), df.shape[1])
    return df
