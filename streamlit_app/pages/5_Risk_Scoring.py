import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from src.model_utils import load_risk_model, get_model_metrics, predict_single_order, FEATURES

st.set_page_config(page_title="Risk Scoring", layout="wide")
st.title("🔴 Late Delivery Risk Scoring")
st.markdown("LightGBM classifier trained on pre-shipment features — no leakers. AUC 0.73 on 36K held-out orders.")
st.markdown("---")

df = st.session_state.get("df")
if df is None:
    st.warning("⚠️ Return to the Home page to load the data first.")
    st.stop()

@st.cache_resource(show_spinner="Loading model...")
def load_model_cached():
    return load_risk_model()

try:
    model = load_model_cached()
except FileNotFoundError:
    st.error("Model file not found. Run notebook 05 to train and save the model first.")
    st.stop()

X = df[FEATURES]
y = df["late_delivery_risk"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
metrics = get_model_metrics(model, X_test, y_test)

tab1, tab2, tab3 = st.tabs(["Model Performance", "What Drives Late Deliveries", "Live Risk Score"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_prob"])
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.4f}",
                            line=dict(color="#4A6FA5", width=2.5))
        fig_roc.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                            line=dict(color="#d1d5db", dash="dash"))
        fig_roc.update_layout(
            title=f"ROC Curve (AUC = {roc_auc:.4f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            plot_bgcolor="white", height=400,
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        cm = metrics["confusion_matrix"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Predicted: On Time", "Predicted: Late"],
            y=["Actual: On Time", "Actual: Late"],
            colorscale="Blues", text=cm, texttemplate="%{text:,}",
            textfont=dict(size=16),
        ))
        fig_cm.update_layout(title="Confusion Matrix", height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

    report = pd.DataFrame(metrics["classification_report"]).T
    report = report[["precision","recall","f1-score","support"]].round(3)
    report["support"] = report["support"].astype(str).str.replace(".0","", regex=False)
    st.dataframe(report.style.format({"precision":"{:.3f}","recall":"{:.3f}","f1-score":"{:.3f}"}),
                 use_container_width=True)

with tab2:
    st.markdown("""
**What the model learned — in plain English:**

- **Scheduled shipping days** is the strongest signal. Orders with longer promised delivery windows slip more —
  carriers treat them as lower priority, and that buffer time gets eroded by handling delays before shipment even starts.

- **Shipping mode** is the second biggest driver. First Class and Second Class orders carry significantly higher
  late risk than their SLA promises suggest. This isn't about cost — it's about network capacity at those service tiers.

- **Market / region** contributes at the order level. Certain markets have structurally tighter carrier options
  which compress the ability to recover from a delay once it starts.

- **Product category** matters because some categories have more complex pick-and-pack requirements —
  large or multi-item orders in categories like Fishing or Outdoors take longer to process, eating into transit time.

- **Order quantity** has a moderate positive effect on late risk. Larger orders are harder to fulfil from a single
  pick location and more likely to be partially split, introducing coordination delays.
""")

    st.markdown("#### SHAP Feature Importance")

    @st.cache_data(show_spinner="Computing SHAP values...")
    def get_shap(_model, X_sample):
        explainer = shap.TreeExplainer(_model)
        raw = explainer.shap_values(X_sample)
        return raw[1] if isinstance(raw, list) else raw

    X_sample = X_test.sample(min(2000, len(X_test)), random_state=42)
    shap_vals = get_shap(model, X_sample)

    fig_s, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_vals, X_sample, plot_type="dot", show=False, color_bar=True)
    fig_s = plt.gcf()
    fig_s.tight_layout()
    st.pyplot(fig_s)
    plt.close()

with tab3:
    st.markdown("Enter order details to get a real-time late delivery risk probability from the trained model.")
    st.markdown("---")

    encoders = df.attrs.get("label_encoders", {})

    def safe_encode(col, val):
        enc = encoders.get(col)
        if enc is None:
            return 0
        classes = list(enc.classes_)
        return classes.index(val) if val in classes else 0

    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)

        mode_sel   = c1.selectbox("Shipping Mode", sorted(df["shipping_mode"].dropna().unique()))
        market_sel = c2.selectbox("Market", sorted(df["market"].dropna().unique()))
        seg_sel    = c3.selectbox("Customer Segment", sorted(df["customer_segment"].dropna().unique()))
        cat_sel    = c1.selectbox("Product Category", sorted(df["category_name"].dropna().unique()))
        qty        = c2.number_input("Order Quantity", min_value=1, max_value=50, value=1)
        sales_val  = c3.number_input("Order Value ($)", min_value=1.0, value=75.0, step=5.0)
        sched_days = c1.number_input("Scheduled Shipping Days", min_value=1, max_value=10, value=4)

        submitted = st.form_submit_button("Calculate Risk →", use_container_width=True)

    if submitted:
        params = {
            "shipping_mode_enc":    safe_encode("shipping_mode", mode_sel),
            "market_enc":           safe_encode("market", market_sel),
            "customer_segment_enc": safe_encode("customer_segment", seg_sel),
            "category_name_enc":    safe_encode("category_name", cat_sel),
            "order_quantity":       qty,
            "sales":                sales_val,
            "days_shipping_scheduled": sched_days,
        }
        prob = predict_single_order(model, params)

        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            color = "#E8735A" if prob >= 0.6 else "#F5B942" if prob >= 0.4 else "#6BAB8A"
            label = "High Risk" if prob >= 0.6 else "Moderate Risk" if prob >= 0.4 else "Low Risk"
            st.markdown(
                f"<div style='text-align:center; padding:2rem; border-radius:12px; background:{color}22; border:2px solid {color}'>"
                f"<p style='font-size:3rem; font-weight:800; color:{color}; margin:0'>{prob:.1%}</p>"
                f"<p style='font-size:1.1rem; color:{color}; margin:0'>{label}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col2:
            if prob >= 0.6:
                st.error(
                    f"This order has a **{prob:.1%} probability of late delivery**. "
                    f"Consider upgrading the shipping mode or flagging for proactive customer communication."
                )
            elif prob >= 0.4:
                st.warning(
                    f"**{prob:.1%} late delivery risk.** Worth monitoring — "
                    f"if the order value justifies it, a mode upgrade would reduce exposure."
                )
            else:
                st.success(
                    f"**{prob:.1%} late delivery risk.** This order looks low-risk given the "
                    f"current shipping mode and scheduled window."
                )
