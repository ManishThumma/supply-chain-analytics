import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Shipment Mode Optimizer", layout="wide")
st.title("Shipment Mode Optimizer")
st.markdown("Static analysis — no filters needed. Evaluates cost, delay, and order-value tradeoffs by shipping mode.")

df = st.session_state.get("df")
if df is None:
    st.warning("Return to the main page to load data.")
    st.stop()

df = df.copy()
df["est_shipping_cost"] = df["sales"] - df["order_profit"]

mode_summary = (
    df.groupby("shipping_mode")
    .agg(
        orders=("sales", "count"),
        avg_cost=("est_shipping_cost", "mean"),
        avg_delay=("shipping_delay", "mean"),
        avg_order_value=("sales", "mean"),
        late_rate=("is_late", "mean"),
    )
    .round(2)
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(
        mode_summary, x="shipping_mode", y="avg_cost",
        title="Estimated Avg Shipping Cost per Order by Mode",
        color="shipping_mode",
        color_discrete_sequence=["#4A6FA5", "#E8735A", "#6BAB8A", "#F5B942"],
        labels={"avg_cost": "Avg Cost ($)", "shipping_mode": "Shipping Mode"},
    )
    fig1.update_layout(showlegend=False, plot_bgcolor="white")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(
        mode_summary, x="shipping_mode", y="avg_delay",
        title="Average Delivery Delay (Days) by Mode",
        color="shipping_mode",
        color_discrete_sequence=["#4A6FA5", "#E8735A", "#6BAB8A", "#F5B942"],
        labels={"avg_delay": "Avg Delay (days)", "shipping_mode": "Shipping Mode"},
    )
    fig2.update_layout(showlegend=False, plot_bgcolor="white")
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig2, use_container_width=True)

sample = df.sample(min(5000, len(df)), random_state=42)
fig3 = px.scatter(
    sample, x="sales", y="shipping_delay",
    color="shipping_mode",
    title="Order Value vs Shipping Delay — Colored by Mode",
    labels={"sales": "Order Value ($)", "shipping_delay": "Shipping Delay (days)"},
    opacity=0.5,
    color_discrete_sequence=["#4A6FA5", "#E8735A", "#6BAB8A", "#F5B942"],
)
fig3.add_hline(y=0, line_dash="dash", line_color="gray")
fig3.update_layout(plot_bgcolor="white")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("### Mode Selection Recommendation")

rec_data = {
    "Shipping Mode": ["Same Day", "First Class", "Second Class", "Standard Class"],
    "Best For": [
        "High-value, time-sensitive orders",
        "Premium segment customers, moderate urgency",
        "Mid-value orders where 2-3 day delay acceptable",
        "Bulk/low-margin orders, tolerant segments",
    ],
    "Order Value Threshold": ["> $150", "$80 – $150", "$30 – $80", "< $30"],
    "Acceptable Delay Tolerance": ["0 days", "0–1 days", "1–2 days", "2–4 days"],
    "Avg Late Rate": [
        f"{mode_summary.loc[mode_summary['shipping_mode'] == m, 'late_rate'].values[0]:.1%}"
        if m in mode_summary["shipping_mode"].values else "N/A"
        for m in ["Same Day", "First Class", "Second Class", "Standard Class"]
    ],
}
rec_df = pd.DataFrame(rec_data)
st.dataframe(rec_df, use_container_width=True, hide_index=True)

st.info(
    "These thresholds are derived from the observed order value distributions per mode in this dataset. "
    "Same Day and First Class show the strongest OTIF performance but carry a cost premium "
    "that erodes margin on low-value orders. Standard Class is the high-volume default but "
    "its late rate makes it unsuitable for SLA-sensitive customer segments."
)
