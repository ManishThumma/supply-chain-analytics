import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.viz_utils import plot_profit_by_segment_region

st.set_page_config(page_title="Profitability", layout="wide")
st.title("Profitability Analysis")

df = st.session_state.get("df")
if df is None:
    st.warning("Return to the main page to load data.")
    st.stop()

st.sidebar.header("Filters")
segments = sorted(df["customer_segment"].dropna().unique())
selected_segments = st.sidebar.multiselect("Customer Segment", segments, default=segments)

markets = sorted(df["market"].dropna().unique())
selected_markets = st.sidebar.multiselect("Market", markets, default=markets)

fdf = df[df["customer_segment"].isin(selected_segments) & df["market"].isin(selected_markets)]

col1, col2 = st.columns(2)

with col1:
    tier_counts = fdf["profit_tier"].value_counts()
    fig1 = px.pie(
        values=tier_counts.values,
        names=tier_counts.index,
        title="Order Distribution by Profit Tier",
        hole=0.4,
        color_discrete_map={
            "Loss": "#E8735A",
            "Breakeven": "#F5B942",
            "Low Margin": "#6BAB8A",
            "Healthy": "#4A6FA5",
        },
    )
    fig1.update_traces(textinfo="percent+label")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = plot_profit_by_segment_region(fdf)
    st.plotly_chart(fig2, use_container_width=True)

cat_profit = (
    fdf.groupby("category_name")["order_profit"]
    .mean()
    .round(2)
    .reset_index()
    .rename(columns={"order_profit": "avg_profit"})
    .sort_values("avg_profit")
)

top10 = cat_profit.tail(10)
bot10 = cat_profit.head(10)

c1, c2 = st.columns(2)
with c1:
    fig3 = px.bar(top10, x="avg_profit", y="category_name", orientation="h",
                  title="Top 10 Categories by Avg Profit",
                  color_discrete_sequence=["#4A6FA5"],
                  labels={"avg_profit": "Avg Profit ($)", "category_name": ""})
    fig3.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)

with c2:
    fig4 = px.bar(bot10, x="avg_profit", y="category_name", orientation="h",
                  title="Bottom 10 Categories by Avg Profit",
                  color_discrete_sequence=["#E8735A"],
                  labels={"avg_profit": "Avg Profit ($)", "category_name": ""})
    fig4.update_layout(plot_bgcolor="white")
    fig4.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig4, use_container_width=True)

st.info(
    "Loss-generating categories aren't automatic candidates for discontinuation — "
    "they may anchor high-margin cross-sell purchases. The more actionable test is "
    "whether those categories are disproportionately assigned to expensive shipping modes. "
    "If so, a mode-shift policy alone may recover margin without touching pricing or range."
)
