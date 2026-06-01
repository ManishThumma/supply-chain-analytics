import sys
from pathlib import Path

# Walk up from this file until we find the repo root (directory containing src/)
_here = Path(__file__).resolve()
for _parent in [_here.parent, _here.parent.parent, _here.parent.parent.parent]:
    if (_parent / "src").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.viz_utils import plot_on_time_rate_by_mode

st.set_page_config(page_title="Delivery Performance", layout="wide")
st.title("📊 Delivery Performance")
st.markdown("On-time vs late breakdown across shipping modes, markets, and geographies.")
st.markdown("---")

df = st.session_state.get("df")
if df is None:
    st.warning("⚠️ Return to the Home page to load the data first.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    modes = sorted(df["shipping_mode"].dropna().unique())
    sel_modes = st.multiselect("Shipping Mode", modes, default=modes)

    markets = sorted(df["market"].dropna().unique())
    sel_markets = st.multiselect("Market", markets, default=markets)

    min_d, max_d = df["order_date"].min().date(), df["order_date"].max().date()
    date_range = st.date_input("Order Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

fdf = df[df["shipping_mode"].isin(sel_modes) & df["market"].isin(sel_markets)]
if len(date_range) == 2:
    fdf = fdf[(fdf["order_date"].dt.date >= date_range[0]) & (fdf["order_date"].dt.date <= date_range[1])]

c1, c2, c3 = st.columns(3)
c1.metric("Orders in Selection", f"{len(fdf):,}")
c2.metric("Late Orders", f"{fdf['is_late'].sum():,}")
c3.metric("Late Rate", f"{fdf['is_late'].mean()*100:.1f}%")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    fig = plot_on_time_rate_by_mode(fdf)
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    monthly = (
        fdf.set_index("order_date").resample("ME")["is_late"]
        .mean().mul(100).reset_index()
        .rename(columns={"is_late": "late_pct"})
    )
    fig2 = px.line(
        monthly, x="order_date", y="late_pct",
        title="Monthly Late Delivery Rate (%)",
        labels={"order_date": "", "late_pct": "Late Rate (%)"},
        color_discrete_sequence=["#4A6FA5"],
    )
    fig2.update_traces(line_width=2.5)
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380)
    st.plotly_chart(fig2, use_container_width=True)

# Market late rate bar
market_stats = (
    fdf.groupby("market")
    .agg(orders=("is_late","count"), late=("is_late","sum"))
    .assign(late_pct=lambda x: (x["late"]/x["orders"]*100).round(1))
    .reset_index().sort_values("late_pct", ascending=True)
)
fig3 = px.bar(
    market_stats, x="late_pct", y="market", orientation="h",
    title="Late Delivery Rate by Market (%)",
    color="late_pct",
    color_continuous_scale=["#4A6FA5","#E8735A"],
    labels={"late_pct": "Late Rate (%)", "market": ""},
    text="late_pct",
)
fig3.update_traces(texttemplate="%{text}%", textposition="outside")
fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", showlegend=False, coloraxis_showscale=False, height=320)
st.plotly_chart(fig3, use_container_width=True)

# Top 10 states
top_states = (
    fdf[fdf["is_late"] == 1]
    .groupby("customer_state").size()
    .sort_values(ascending=True).tail(10)
    .reset_index(name="late_count")
)
fig4 = px.bar(
    top_states, x="late_count", y="customer_state", orientation="h",
    title="Top 10 States by Late Delivery Count",
    color_discrete_sequence=["#E8735A"],
    labels={"late_count": "Late Orders", "customer_state": ""},
)
fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=340)
st.plotly_chart(fig4, use_container_width=True)

worst_mode = fdf.groupby("shipping_mode")["is_late"].mean().idxmax()
best_mode  = fdf.groupby("shipping_mode")["is_late"].mean().idxmin()
worst_market = fdf.groupby("market")["is_late"].mean().idxmax()
market_range = fdf.groupby("market")["is_late"].mean()
market_spread = (market_range.max() - market_range.min()) * 100

st.info(
    f"**Takeaway:** {fdf['is_late'].mean():.1%} of orders in this selection were late. "
    f"{worst_mode} has the highest late rate while {best_mode} performs best. "
    f"Across markets, the spread between best and worst is only {market_spread:.1f} percentage points — "
    f"{worst_market} leads in late rate but the consistency across markets points to a network-wide SLA issue "
    f"rather than a regional carrier problem."
)
