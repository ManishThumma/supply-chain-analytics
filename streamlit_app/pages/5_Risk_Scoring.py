import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.feature_engineering import compute_delivery_delta, flag_late_orders, encode_categoricals
from src.model_utils import load_risk_model, get_model_metrics, predict_single_order, FEATURES
from src.viz_utils import plot_shap_summary

st.set_page_config(page_title="Risk Scoring", layout="wide")
st.title("Late Delivery Risk Scoring")

df = st.session_state.get("df")
if df is None:
    st.warning("Return to the main page to load data.")
    st.stop()

try:
    model = load_risk_model()
except FileNotFoundError:
    st.error("Model not found. Run notebook 05 to train and save the model first.")
    st.stop()

X = df[FEATURES]
y = df["late_delivery_risk"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
metrics = get_model_metrics(model, X_test, y_test)

st.markdown("## Section A: Model Performance")

fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_prob"])
roc_auc = auc(fpr, tpr)

col1, col2 = st.columns(2)

with col1:
    fig_roc = go.Figure()
    fig_roc.add_scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.4f}",
                        line=dict(color="#4A6FA5", width=2))
    fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                        line=dict(color="gray", dash="dash"))
    fig_roc.update_layout(
        title="ROC Curve — Late Delivery Risk Classifier",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col2:
    cm = metrics["confusion_matrix"]
    fig_cm = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Predicted: On Time", "Predicted: Late"],
            y=["Actual: On Time", "Actual: Late"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
        )
    )
    fig_cm.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

report_df = pd.DataFrame(metrics["classification_report"]).T
st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

st.markdown("---")
st.markdown("## Section B: What Drives Late Deliveries")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

fig_shap = plot_shap_summary(shap_values, X_test)
st.pyplot(fig_shap)

st.markdown(
    """
**Key drivers of late delivery risk — plain English summary:**

- **Shipping mode** is the single strongest predictor. Orders routed through Standard Class carry significantly higher late delivery risk compared to Same Day or First Class. This reflects carrier network capacity constraints rather than geographic factors.
- **Scheduled delivery days** amplifies risk for orders with longer promised windows. Carriers appear to treat longer lead-time orders as lower priority, resulting in more slippage against SLAs.
- **Market region** matters — certain international markets (particularly Latin America and parts of Asia) show structural late delivery risk that persists regardless of shipping mode, pointing to last-mile infrastructure gaps.
- **Order quantity** has a moderate positive correlation with late risk. Larger orders may be harder to fulfil from a single pick location, introducing handling delays that cascade into shipping delays.
- **Customer segment** contributes at the margin — Home Office orders show slightly different risk profiles than Consumer or Corporate, likely reflecting differences in fulfilment priority rules.
"""
)

st.markdown("---")
st.markdown("## Live Risk Scoring")
st.markdown("Enter order parameters below to get a real-time late delivery risk probability.")

with st.form("risk_score_form"):
    c1, c2, c3 = st.columns(3)

    shipping_mode_options = sorted(df["shipping_mode"].dropna().unique())
    market_options = sorted(df["market"].dropna().unique())
    segment_options = sorted(df["customer_segment"].dropna().unique())
    category_options = sorted(df["category_name"].dropna().unique())

    shipping_mode_sel = c1.selectbox("Shipping Mode", shipping_mode_options)
    market_sel = c2.selectbox("Market", market_options)
    segment_sel = c3.selectbox("Customer Segment", segment_options)
    category_sel = c1.selectbox("Category", category_options)
    qty = c2.number_input("Order Quantity", min_value=1, max_value=100, value=1)
    sales_val = c3.number_input("Order Value ($)", min_value=1.0, value=50.0, step=5.0)
    sched_days = c1.number_input("Scheduled Shipping Days", min_value=1, max_value=10, value=3)

    submitted = st.form_submit_button("Calculate Risk Score")

if submitted:
    encoders = df.attrs.get("label_encoders", {})

    def safe_encode(enc, val):
        classes = list(enc.classes_)
        return classes.index(val) if val in classes else 0

    order_params = {
        "shipping_mode_enc": safe_encode(encoders["shipping_mode"], shipping_mode_sel),
        "market_enc": safe_encode(encoders["market"], market_sel),
        "customer_segment_enc": safe_encode(encoders["customer_segment"], segment_sel),
        "category_name_enc": safe_encode(encoders["category_name"], category_sel),
        "order_quantity": qty,
        "sales": sales_val,
        "days_shipping_scheduled": sched_days,
    }

    risk_prob = predict_single_order(model, order_params)

    if risk_prob >= 0.7:
        st.error(f"**Late Delivery Risk: {risk_prob:.1%}** — High Risk. Consider upgrading shipping mode.")
    elif risk_prob >= 0.4:
        st.warning(f"**Late Delivery Risk: {risk_prob:.1%}** — Moderate Risk. Monitor closely.")
    else:
        st.success(f"**Late Delivery Risk: {risk_prob:.1%}** — Low Risk.")
