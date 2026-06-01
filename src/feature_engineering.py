import pandas as pd
from sklearn.preprocessing import LabelEncoder


def compute_delivery_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["shipping_delay"] = df["days_shipping_real"] - df["days_shipping_scheduled"]
    return df


def flag_late_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Delivery Status and Late_delivery_risk are both present in the dataset.
    # We cross-validate: an order is flagged late only when both signals agree.
    # This removes ambiguous cases (e.g., "Shipping on Time" with risk=1).
    status_late = df["delivery_status"].str.contains("Late", case=False, na=False)
    df["is_late"] = ((df["late_delivery_risk"] == 1) & status_late).astype(int)
    return df


def compute_order_profitability_tier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [-float("inf"), 0, 5, 20, float("inf")]
    labels = ["Loss", "Breakeven", "Low Margin", "Healthy"]
    df["profit_tier"] = pd.cut(df["order_profit"], bins=bins, labels=labels)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    encode_cols = ["shipping_mode", "market", "customer_segment", "category_name"]
    encoders = {}
    for col in encode_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    df.attrs["label_encoders"] = encoders
    return df
