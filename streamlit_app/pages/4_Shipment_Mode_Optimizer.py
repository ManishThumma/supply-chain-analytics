import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Shipment Mode Optimizer", layout="wide")
st.title("🚚 Shipment Mode Optimizer")
st.markdown("Cost, delay, and value tradeoffs across shipping modes — no filters needed.")
st.markdown("---")

df = st.session_state.get("df")
if df is None:
    st.warning("⚠️ Return to the Home page to load the data first.")
    st.stop()

dfc = df.copy()
dfc["est_cost"] = dfc["sales"] - dfc["order_profit"]

mode_summary = (
    dfc.groupby("shipping_mode")
    .agg(
        orders=("sales", "count"),
        avg_cost=("est_cost", "mean"),
        avg_delay=("shipping_delay", "mean"),
        avg_value=("sales", "mean"),
        late_rate=("is_late", "mean"),
    )
    .round(3).reset_index()
)

col1, col2, col3 = st.columns(3)

with col1:
    fig1 = px.bar(
        mode_summary.sort_values("avg_cost"),
        x="shipping_mode", y="avg_cost",
        title="Avg Estimated Shipping Cost per Order",
        color="shipping_mode",
        color_discrete_sequence=["#4A6FA5","#6BAB8A","#F5B942","#E8735A"],
        labels={"avg_cost": "Avg Cost ($)", "shipping_mode": ""},
        text="avg_cost",
    )
    fig1.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
    fig1.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", height=360)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(
        mode_summary.sort_values("avg_delay"),
        x="shipping_mode", y="avg_delay",
        title="Avg Delivery Delay (days)",
        color="shipping_mode",
        color_discrete_sequence=["#4A6FA5","#6BAB8A","#F5B942","#E8735A"],
        labels={"avg_delay": "Avg Delay (days)", "shipping_mode": ""},
        text="avg_delay",
    )
    fig2.update_traces(texttemplate="%{text:.2f}d", textposition="outside")
    fig2.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", height=360)
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    fig3 = px.bar(
        mode_summary.sort_values("late_rate"),
        x="shipping_mode", y=mode_summary["late_rate"].mul(100).round(1),
        title="Late Delivery Rate (%)",
        color="shipping_mode",
        color_discrete_sequence=["#4A6FA5","#6BAB8A","#F5B942","#E8735A"],
        labels={"y": "Late Rate (%)", "shipping_mode": ""},
        text=mode_summary["late_rate"].mul(100).round(1),
    )
    fig3.update_traces(texttemplate="%{text}%", textposition="outside")
    fig3.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", height=360)
    st.plotly_chart(fig3, use_container_width=True)

sample = dfc.sample(min(8000, len(dfc)), random_state=42)
fig4 = px.scatter(
    sample, x="sales", y="shipping_delay",
    color="shipping_mode",
    title="Order Value vs Shipping Delay by Mode",
    labels={"sales": "Order Value ($)", "shipping_delay": "Shipping Delay (days)"},
    opacity=0.4,
    color_discrete_sequence=["#4A6FA5","#E8735A","#6BAB8A","#F5B942"],
)
fig4.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("### When to Use Each Mode")
mode_rows = []
for _, row in mode_summary.iterrows():
    late_pct = f"{row['late_rate']*100:.1f}%"
    mode_rows.append({
        "Mode": row["shipping_mode"],
        "Best For": {
            "Same Day": "High-value, time-sensitive orders (>$150)",
            "First Class": "Premium customers, moderate urgency ($80–150)",
            "Second Class": "Mid-value orders, 1–2 day delay acceptable ($30–80)",
            "Standard Class": "High-volume, low-margin orders (<$30)",
        }.get(row["shipping_mode"], "—"),
        "Avg Order Value": f"${row['avg_value']:.2f}",
        "Avg Delay": f"{row['avg_delay']:.2f} days",
        "Late Rate": late_pct,
        "Est. Shipping Cost": f"${row['avg_cost']:.2f}",
    })

st.dataframe(pd.DataFrame(mode_rows), use_container_width=True, hide_index=True)

cheapest = mode_summary.loc[mode_summary["avg_cost"].idxmin(), "shipping_mode"]
most_reliable = mode_summary.loc[mode_summary["late_rate"].idxmin(), "shipping_mode"]
most_reliable_rate = mode_summary["late_rate"].min()
highest_value_mode = mode_summary.loc[mode_summary["avg_value"].idxmax(), "shipping_mode"]
highest_value = mode_summary["avg_value"].max()

st.info(
    f"**Takeaway:** {cheapest} has the lowest estimated shipping cost. "
    f"{most_reliable} has the best on-time rate at {most_reliable_rate:.1%}. "
    f"{highest_value_mode} carries the highest average order value (${highest_value:.2f}), "
    f"which justifies its cost premium. The core question is matching mode to order value — "
    f"putting a low-margin order on an expensive mode wipes the profit entirely."
)
