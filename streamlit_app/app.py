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
from src.data_loader import load_data
from src.feature_engineering import (
    compute_delivery_delta,
    flag_late_orders,
    compute_order_profitability_tier,
    encode_categoricals,
    add_predictive_features,
)

st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #6b7280; }
.block-container { padding-top: 2rem; }
h1 { font-weight: 800; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading 180K orders...")
def get_data():
    df = load_data()
    df = compute_delivery_delta(df)
    df = flag_late_orders(df)
    df = compute_order_profitability_tier(df)
    df = encode_categoricals(df)
    df = add_predictive_features(df)
    return df


df = get_data()
st.session_state["df"] = df

st.title("📦 Supply Chain Performance Analytics")
st.markdown("**DataCo Global Operations · 180,519 Orders · Jan 2015 – Jan 2018**")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Orders", f"{len(df):,}")
c2.metric("On-Time Rate", f"{(1 - df['is_late'].mean()) * 100:.1f}%")
c3.metric("Avg Profit / Order", f"${df['order_profit'].mean():.2f}")
c4.metric("Late Delivery Risk", f"{df['late_delivery_risk'].mean() * 100:.1f}%")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
### About This Project

This dashboard analyses operational performance across the DataCo global supply chain —
covering five markets, four shipping modes, and three years of order history.
The goal was to go beyond surface-level reporting and answer the questions
that actually matter in supply chain operations: where are deliveries failing,
which categories are quietly losing money, and can we predict late shipments
before they leave the warehouse?

The risk scoring page includes a trained **LightGBM classifier** (AUC 0.73) that
predicts late delivery probability from pre-shipment order attributes only —
no post-hoc leakers — so the output is genuinely actionable at fulfilment time.
""")

with col2:
    st.markdown("### Navigate")
    st.markdown("""
- 📊 **Delivery Performance** — OTIF rates by mode & market
- 📈 **Demand Trends** — Volume & seasonality by department
- 💰 **Profitability** — Margin by segment, region & category
- 🚚 **Shipment Mode Optimizer** — Cost vs delay tradeoffs
- 🔴 **Risk Scoring** — Live ML predictions per order
""")

st.markdown("---")
st.markdown(
    "<p style='color:#9ca3af;font-size:0.8rem'>Dataset: "
    "<a href='https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis' "
    "target='_blank'>DataCo Smart Supply Chain</a> · "
    "Built by <a href='https://github.com/ManishThumma' target='_blank'>Manish Thumma</a></p>",
    unsafe_allow_html=True
)
