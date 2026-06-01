import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Demand Trends", layout="wide")
st.title("📈 Demand Trends")
st.markdown("Order volume patterns, seasonality, and year-over-year growth by department and category.")
st.markdown("---")

df = st.session_state.get("df")
if df is None:
    st.warning("⚠️ Return to the Home page to load the data first.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    depts = sorted(df["department_name"].dropna().unique())
    sel_depts = st.multiselect("Department", depts, default=depts)

    markets = sorted(df["market"].dropna().unique())
    sel_markets = st.multiselect("Market", markets, default=markets)

    min_d, max_d = df["order_date"].min().date(), df["order_date"].max().date()
    date_range = st.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

fdf = df[df["department_name"].isin(sel_depts) & df["market"].isin(sel_markets)].copy()
if len(date_range) == 2:
    fdf = fdf[(fdf["order_date"].dt.date >= date_range[0]) & (fdf["order_date"].dt.date <= date_range[1])]

# Rolling volume chart
daily = (
    fdf.set_index("order_date").resample("D")["order_quantity"].sum()
    .reset_index()
)
daily["ma30"] = daily["order_quantity"].rolling(30).mean()
daily["ma90"] = daily["order_quantity"].rolling(90).mean()

fig1 = go.Figure()
fig1.add_scatter(x=daily["order_date"], y=daily["order_quantity"], mode="lines",
                 name="Daily", line=dict(color="#4A6FA5", width=1), opacity=0.3)
fig1.add_scatter(x=daily["order_date"], y=daily["ma30"], mode="lines",
                 name="30-Day MA", line=dict(color="#E8735A", width=2.5))
fig1.add_scatter(x=daily["order_date"], y=daily["ma90"], mode="lines",
                 name="90-Day MA", line=dict(color="#6BAB8A", width=2.5))
fig1.update_layout(title="Daily Order Volume with Rolling Averages",
                   yaxis_title="Units Ordered", plot_bgcolor="rgba(0,0,0,0)", height=380,
                   legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig1, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    cat_heat = (
        fdf.assign(month=fdf["order_date"].dt.month)
        .groupby(["category_name", "month"])["order_quantity"].sum()
        .unstack("month").fillna(0)
    )
    top_cats = cat_heat.sum(axis=1).nlargest(15).index
    cat_heat = cat_heat.loc[top_cats]
    fig2 = px.imshow(
        cat_heat, title="Order Volume: Category × Month (Top 15)",
        labels={"x": "Month", "y": "", "color": "Units"},
        color_continuous_scale="YlOrRd", aspect="auto",
    )
    fig2.update_layout(height=480)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fdf["year"] = fdf["order_date"].dt.year
    yoy = fdf.groupby(["department_name", "year"])["order_quantity"].sum().unstack("year")
    years = sorted(yoy.columns)
    if len(years) >= 2:
        yoy["growth"] = ((yoy[years[-1]] - yoy[years[-2]]) / yoy[years[-2]] * 100).round(1)
        yoy_plot = yoy["growth"].dropna().sort_values().reset_index()
        fig3 = px.bar(
            yoy_plot, x="growth", y="department_name", orientation="h",
            title="YoY Order Growth by Department (%)",
            color="growth",
            color_continuous_scale=["#E8735A", "#f5f5f5", "#4A6FA5"],
            color_continuous_midpoint=0,
            labels={"growth": "YoY Growth (%)", "department_name": ""},
        )
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                           coloraxis_showscale=False, height=480)
        fig3.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
        st.plotly_chart(fig3, use_container_width=True)

st.info(
    "**Takeaway:** Fan Shop and Golf dominate order volume. Categories with Q4 spikes (months 10–12) "
    "should be flagged for early inventory pre-positioning. Departments with negative YoY figures "
    "may reflect partial-year data rather than structural decline — worth verifying coverage before acting on it."
)
