import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.viz_utils import plot_profit_by_segment_region

st.set_page_config(page_title="Profitability", layout="wide")
st.title("💰 Profitability Analysis")
st.markdown("Order-level margin breakdown by profit tier, customer segment, market, and product category.")
st.markdown("---")

df = st.session_state.get("df")
if df is None:
    st.warning("⚠️ Return to the Home page to load the data first.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    segs = sorted(df["customer_segment"].dropna().unique())
    sel_segs = st.multiselect("Customer Segment", segs, default=segs)
    markets = sorted(df["market"].dropna().unique())
    sel_markets = st.multiselect("Market", markets, default=markets)

fdf = df[df["customer_segment"].isin(sel_segs) & df["market"].isin(sel_markets)]

c1, c2, c3 = st.columns(3)
c1.metric("Avg Profit / Order", f"${fdf['order_profit'].mean():.2f}")
c2.metric("Loss-Making Orders", f"{(fdf['order_profit'] < 0).mean():.1%}")
c3.metric("Healthy Margin Orders", f"{(fdf['profit_tier'] == 'Healthy').mean():.1%}")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    tier_counts = fdf["profit_tier"].value_counts()
    fig1 = px.pie(
        values=tier_counts.values, names=tier_counts.index,
        title="Order Distribution by Profit Tier",
        hole=0.45,
        color_discrete_map={
            "Loss": "#E8735A", "Breakeven": "#F5B942",
            "Low Margin": "#6BAB8A", "Healthy": "#4A6FA5",
        },
    )
    fig1.update_traces(textinfo="percent+label", pull=[0.03]*len(tier_counts))
    fig1.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = plot_profit_by_segment_region(fdf)
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)

cat_profit = (
    fdf.groupby("category_name")["order_profit"]
    .mean().round(2).reset_index()
    .rename(columns={"order_profit": "avg_profit"})
    .sort_values("avg_profit")
)

c1, c2 = st.columns(2)
with c1:
    top10 = cat_profit.tail(10)
    fig3 = px.bar(top10, x="avg_profit", y="category_name", orientation="h",
                  title="Top 10 Categories — Avg Profit / Order",
                  color_discrete_sequence=["#4A6FA5"],
                  labels={"avg_profit": "Avg Profit ($)", "category_name": ""},
                  text="avg_profit")
    fig3.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380)
    st.plotly_chart(fig3, use_container_width=True)

with c2:
    bot10 = cat_profit.head(10)
    fig4 = px.bar(bot10, x="avg_profit", y="category_name", orientation="h",
                  title="Bottom 10 Categories — Avg Profit / Order",
                  color_discrete_sequence=["#E8735A"],
                  labels={"avg_profit": "Avg Profit ($)", "category_name": ""},
                  text="avg_profit")
    fig4.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
    fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380)
    fig4.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig4, use_container_width=True)

best_cat  = cat_profit.iloc[-1]
worst_cat = cat_profit.iloc[0]
loss_pct  = (fdf["order_profit"] < 0).mean()
avg_profit = fdf["order_profit"].mean()

st.info(
    f"**Takeaway:** Average profit per order in this selection is **${avg_profit:.2f}**, with {loss_pct:.1%} of orders loss-generating. "
    f"{best_cat['category_name']} leads at ${best_cat['avg_profit']:.2f} avg profit per order; "
    f"{worst_cat['category_name']} sits at the bottom at ${worst_cat['avg_profit']:.2f}. "
    f"Before cutting low-margin categories, check whether they're being shipped via expensive modes — "
    f"a mode-shift policy alone may recover margin without touching pricing or range."
)
