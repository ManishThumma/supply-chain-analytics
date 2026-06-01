import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Demand Trends", layout="wide")
st.title("Demand Trends")

df = st.session_state.get("df")
if df is None:
    st.warning("Return to the main page to load data.")
    st.stop()

st.sidebar.header("Filters")

departments = sorted(df["department_name"].dropna().unique())
selected_depts = st.sidebar.multiselect("Department", departments, default=departments)

markets = sorted(df["market"].dropna().unique())
selected_markets = st.sidebar.multiselect("Market", markets, default=markets)

min_date = df["order_date"].min().date()
max_date = df["order_date"].max().date()
date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

fdf = df[df["department_name"].isin(selected_depts) & df["market"].isin(selected_markets)]
if len(date_range) == 2:
    fdf = fdf[(fdf["order_date"].dt.date >= date_range[0]) & (fdf["order_date"].dt.date <= date_range[1])]

daily = (
    fdf.set_index("order_date")
    .resample("D")["order_quantity"]
    .sum()
    .reset_index()
)
daily["ma30"] = daily["order_quantity"].rolling(30).mean()
daily["ma90"] = daily["order_quantity"].rolling(90).mean()

fig1 = go.Figure()
fig1.add_scatter(x=daily["order_date"], y=daily["order_quantity"], mode="lines",
                 name="Daily", line=dict(color="#4A6FA5", width=1), opacity=0.35)
fig1.add_scatter(x=daily["order_date"], y=daily["ma30"], mode="lines",
                 name="30-Day MA", line=dict(color="#E8735A", width=2))
fig1.add_scatter(x=daily["order_date"], y=daily["ma90"], mode="lines",
                 name="90-Day MA", line=dict(color="#6BAB8A", width=2))
fig1.update_layout(title="Daily Order Volume with Rolling Averages",
                   yaxis_title="Units", plot_bgcolor="white")
st.plotly_chart(fig1, use_container_width=True)

cat_heatmap = (
    fdf.assign(month=fdf["order_date"].dt.month)
    .groupby(["category_name", "month"])["order_quantity"]
    .sum()
    .unstack("month")
    .fillna(0)
)
top_cats = cat_heatmap.sum(axis=1).nlargest(15).index
cat_heatmap = cat_heatmap.loc[top_cats]

fig2 = px.imshow(
    cat_heatmap,
    title="Order Quantity by Category × Month (Top 15)",
    labels={"x": "Month", "y": "Category", "color": "Units"},
    color_continuous_scale="YlOrRd",
    aspect="auto",
)
st.plotly_chart(fig2, use_container_width=True)

fdf = fdf.copy()
fdf["year"] = fdf["order_date"].dt.year
yoy = (
    fdf.groupby(["department_name", "year"])["order_quantity"]
    .sum()
    .unstack("year")
)
years = sorted(yoy.columns)
if len(years) >= 2:
    yoy["growth_pct"] = ((yoy[years[-1]] - yoy[years[-2]]) / yoy[years[-2]] * 100).round(1)
    yoy_sorted = yoy["growth_pct"].sort_values().reset_index()
    colors = ["#E8735A" if v < 0 else "#4A6FA5" for v in yoy_sorted["growth_pct"]]
    fig3 = px.bar(yoy_sorted, x="growth_pct", y="department_name", orientation="h",
                  title="YoY Order Growth by Department (%)",
                  labels={"growth_pct": "YoY Growth (%)", "department_name": ""},
                  color="growth_pct",
                  color_continuous_scale=["#E8735A", "#FFFFFF", "#4A6FA5"],
                  color_continuous_midpoint=0)
    fig3.update_layout(plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

st.info(
    "Categories with strong month-over-month concentration in Q4 (months 10–12) should be "
    "flagged for early inventory pre-positioning and carrier capacity locking. "
    "Departments showing negative YoY growth warrant a data coverage check before assuming "
    "structural decline — partial-year data in the most recent period mechanically depresses the metric."
)
