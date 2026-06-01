import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.viz_utils import plot_on_time_rate_by_mode

st.set_page_config(page_title="Delivery Performance", layout="wide")
st.title("Delivery Performance Overview")

df = st.session_state.get("df")
if df is None:
    st.warning("Return to the main page to load data.")
    st.stop()

st.sidebar.header("Filters")

shipping_modes = sorted(df["shipping_mode"].dropna().unique())
selected_modes = st.sidebar.multiselect("Shipping Mode", shipping_modes, default=shipping_modes)

markets = sorted(df["market"].dropna().unique())
selected_markets = st.sidebar.multiselect("Market", markets, default=markets)

min_date = df["order_date"].min().date()
max_date = df["order_date"].max().date()
date_range = st.sidebar.date_input("Order Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

fdf = df[
    df["shipping_mode"].isin(selected_modes) &
    df["market"].isin(selected_markets)
]
if len(date_range) == 2:
    fdf = fdf[(fdf["order_date"].dt.date >= date_range[0]) & (fdf["order_date"].dt.date <= date_range[1])]

st.markdown(f"**{len(fdf):,} orders** in current filter selection")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    fig = plot_on_time_rate_by_mode(fdf)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    monthly_late = (
        fdf.set_index("order_date")
        .resample("M")["is_late"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"is_late": "late_rate_pct"})
    )
    fig2 = px.line(
        monthly_late, x="order_date", y="late_rate_pct",
        title="Monthly Late Delivery Rate (%)",
        labels={"order_date": "", "late_rate_pct": "Late Rate (%)"},
        color_discrete_sequence=["#4A6FA5"]
    )
    fig2.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

top_states = (
    fdf[fdf["is_late"] == 1]
    .groupby("customer_state")["is_late"]
    .count()
    .sort_values(ascending=True)
    .tail(10)
    .reset_index()
    .rename(columns={"is_late": "late_count"})
)
fig3 = px.bar(
    top_states, x="late_count", y="customer_state",
    orientation="h",
    title="Top 10 States by Late Delivery Count",
    labels={"late_count": "Late Orders", "customer_state": ""},
    color_discrete_sequence=["#E8735A"]
)
fig3.update_layout(plot_bgcolor="white")
st.plotly_chart(fig3, use_container_width=True)

st.info(
    f"Within the current filter selection, the overall late delivery rate is "
    f"**{fdf['is_late'].mean():.1%}**. "
    f"Standard Class accounts for the largest share of late deliveries by volume, "
    f"while Same Day shipping shows the most consistent on-time performance. "
    f"Geographic concentration of delays varies significantly by market — "
    f"adjusting the Market filter above reveals where carrier network gaps are most acute."
)
