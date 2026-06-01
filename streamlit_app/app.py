import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.data_loader import load_data
from src.feature_engineering import (
    compute_delivery_delta,
    flag_late_orders,
    compute_order_profitability_tier,
    encode_categoricals,
)

st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_data():
    df = load_data(Path(__file__).parent.parent / "data" / "dataco_supply_chain.csv")
    df = compute_delivery_delta(df)
    df = flag_late_orders(df)
    df = compute_order_profitability_tier(df)
    df = encode_categoricals(df)
    return df


df = get_data()
st.session_state["df"] = df

st.title("Supply Chain Performance Analytics")
st.markdown("**DataCo Global Operations Dataset — 180K+ Orders**")
st.markdown("---")

total_orders = len(df)
on_time_rate = (1 - df["is_late"].mean()) * 100
avg_profit = df["order_profit"].mean()
late_risk_rate = df["late_delivery_risk"].mean() * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Orders", f"{total_orders:,}")
c2.metric("On-Time Rate", f"{on_time_rate:.1f}%")
c3.metric("Avg Profit / Order", f"${avg_profit:.2f}")
c4.metric("Late Delivery Risk Rate", f"{late_risk_rate:.1f}%")

st.markdown("---")
st.markdown(
    """
This dashboard analyses operational performance across the DataCo global supply chain dataset,
covering order fulfilment, delivery timeliness, demand patterns, and order-level profitability.
The risk scoring page includes a trained LightGBM classifier that predicts late delivery probability
from pre-shipment order attributes, enabling proactive intervention before a shipment leaves the warehouse.
Use the sidebar to navigate between the five analysis sections.
"""
)

st.sidebar.title("Navigation")
st.sidebar.markdown(
    """
- **Delivery Performance** — OTIF rates by mode and market
- **Demand Trends** — Volume and seasonality by department
- **Profitability** — Margin analysis by segment and category
- **Shipment Mode Optimizer** — Mode cost and delay tradeoffs
- **Risk Scoring** — ML model for late delivery prediction
"""
)
