# Supply Chain Performance Analytics
### Delivery, Demand & Profitability Intelligence across 180K+ Global Orders

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://manish-supply-chain-analytics.streamlit.app)

---

## The Business Problem

Late deliveries and poor margin visibility are two of the most expensive and preventable problems in retail supply chain operations. For companies managing millions of orders across multiple markets and carrier modes, the challenge isn't collecting the data — it's knowing where to look and what decisions to make from it.

This project tackles that across three operational areas: delivery performance, order profitability, and late delivery risk prediction — using three years of real order data from the DataCo global supply chain.

---

## The Dataset

**DataCo Smart Supply Chain** — 180,519 orders across 5 global markets (Europe, LATAM, Pacific Asia, USCA, Africa), spanning January 2015 to January 2018. Covers shipping mode, delivery status, customer segment, product category, order financials, and geography across 53 columns.

Source: [Kaggle — DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

---

## What Was Built

An end-to-end analytics pipeline in Python — data cleaning, feature engineering, exploratory analysis across five Jupyter notebooks, and a five-page interactive Streamlit dashboard backed by a trained late delivery risk classifier.

**Notebooks**
- `01` — Data quality assessment, null handling, cross-validating the two late-delivery signals in the dataset
- `02` — Delivery performance breakdown by shipping mode, market, and geography
- `03` — Demand trend analysis with rolling averages, category seasonality, and YoY growth by department
- `04` — Order-level profitability by tier, segment, market, and product category
- `05` — Late delivery risk model: LightGBM classifier, SHAP feature attribution, AUC-ROC evaluation

**Dashboard Pages**
- Delivery Performance — OTIF rates by mode, market, and state with date and market filters
- Demand Trends — Rolling order volume, category × month heatmap, YoY department growth
- Profitability — Profit tier distribution, segment × market heatmap, top and bottom categories by margin
- Shipment Mode Optimizer — Cost vs delay tradeoffs by mode with a structured mode selection guide
- Risk Scoring — Model performance, SHAP-driven feature importance, and live per-order risk scoring

---

## Approach

The data pipeline is built in `src/` — a `data_loader` that handles encoding, date parsing, and PII removal; a `feature_engineering` module that computes delivery delta, cross-validates the late delivery flag, bins profit tiers, and label-encodes categoricals for modelling; and a `viz_utils` layer of reusable Plotly and Matplotlib functions used across both notebooks and the dashboard.

The risk model uses **LightGBM** trained on pre-shipment features only — shipping mode, market, customer segment, product category, order quantity, order value, and scheduled delivery days. Actual shipping delay is excluded as a feature since it isn't known at fulfilment time. **SHAP TreeExplainer** is used to interpret what the model learned and why.

---

## Findings

**Delivery Performance**

54.8% of orders were late — and the rate barely moves across markets (54–55% across all five). That consistency is the key finding: this isn't a regional carrier problem or a last-mile geography issue, it's a company-wide SLA commitment that outpaces what the network can actually deliver. Standard Class, which carries the highest order volume, had the lowest late rate at 38.1%. First Class and Second Class significantly underperformed their premium positioning.

**Profitability**

Average profit per order was $21.97, but 19.4% of orders were loss-generating and another 3.8% broke even — meaning roughly 1 in 4 orders either cost money or recovered nothing. Computers averaged $157.59 profit per order, the highest category by a wide margin. CDs, Toys, and Books sat under $2.20. Profit variation across customer segments was minimal — category and shipping mode are the real margin drivers, not who's buying.

**Late Delivery Risk Model**

AUC of 0.7367 on 36,104 held-out orders using pre-shipment features only. Scheduled delivery days and shipping mode were the two strongest predictors by SHAP value. The finding: carriers appear to treat longer lead-time commitments as lower priority, so the buffer time built into longer scheduled windows gets eroded before the shipment even moves. The model is deployed in the dashboard's Risk Scoring page for live per-order prediction.

---

## Tech Stack

Python · pandas · LightGBM · SHAP · scikit-learn · Plotly · Streamlit · Matplotlib · Seaborn

---

## About

**Manish Thumma** — Business Data Analyst

- LinkedIn: [linkedin.com/in/balamanishreddythumma](https://www.linkedin.com/in/balamanishreddythumma)
