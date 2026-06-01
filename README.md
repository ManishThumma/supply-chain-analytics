# Supply Chain Performance Analytics
### Delivery, Demand & Profitability Intelligence across 180K+ Global Orders

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://supply-chain-analytics.streamlit.app)

---

## Overview

Late deliveries, margin-negative orders, and misaligned shipping mode decisions are among the most costly and preventable problems in retail supply chain operations. This project analyses 180,519 orders from the DataCo global supply chain dataset to surface where those problems concentrate, what's driving them, and what operations and commercial teams can do about it.

The output is a five-page interactive dashboard built in Streamlit, backed by a trained late-delivery risk model that scores new orders in real time using only information available at the point of fulfilment — before a shipment leaves the warehouse.

---

## Business Questions

- **Where are deliveries failing?** Which shipping modes, markets, and geographies have the worst on-time performance, and is it a carrier problem or a systemic SLA issue?
- **What does order profitability actually look like?** Which product categories are generating margin, and which are quietly eroding it?
- **Can late deliveries be predicted before they happen?** What order attributes at fulfilment time are most predictive of a late shipment?
- **Are we using the right shipping mode for each order?** Is there a mismatch between order value, shipping cost, and mode selection that's compressing margin?

---

## Key Findings

### Delivery Performance
- **54.8% of orders were late** — consistent across all five markets (Europe, LATAM, Pacific Asia, USCA, Africa) within a 1% band. That consistency rules out a regional carrier problem. It points to a company-wide SLA commitment that outpaces actual network capacity.
- **Standard Class had the lowest late rate (38.1%)** despite carrying the highest order volume. First Class and Second Class underperformed their premium positioning significantly — the SLA windows promised to customers don't reflect what the carrier network can reliably deliver.
- California, Puerto Rico, and Illinois account for the highest late order volumes in absolute terms.

### Profitability
- **Average order profit: $21.97** — but nearly **1 in 4 orders either lost money or broke even** (19.4% loss, 3.8% breakeven).
- **Computers** generate $157.59 average profit per order — the highest by a wide margin. Garden, Crafts, Cameras, and Fishing follow.
- **CDs, Toys, and Books** average under $2.20 profit per order. The key question isn't whether to cut them — it's whether they're being shipped via unnecessarily expensive modes that are eroding what little margin they carry.
- Profit variation across Customer Segments (Consumer, Corporate, Home Office) is minimal. Category and shipping mode drive margin — not customer type.

### Shipping Mode Efficiency
- Same Day and First Class carry the highest estimated shipping costs but don't proportionally reduce late rates enough to justify blanket use.
- Standard Class is cost-efficient at scale but unsuitable for time-sensitive or high-value orders where a late delivery creates a customer service cost that exceeds the shipping savings.
- There's a clear opportunity to build an order-value-based mode assignment policy that reduces both shipping cost and late delivery exposure simultaneously.

### Late Delivery Risk Model
- A predictive model trained on pre-shipment features only (shipping mode, market, customer segment, product category, order quantity, order value, scheduled delivery days) achieves an **AUC of 0.73** on 36K held-out orders.
- **Scheduled delivery days and shipping mode** are the two strongest drivers. Carriers treat longer lead-time orders as lower priority — the buffer time that was supposed to protect SLA adherence gets eroded before the shipment is even picked.
- The model is integrated into the dashboard's Risk Scoring page, where you can input any order's attributes and get a real-time late delivery probability score.

---

## Dashboard

Five pages covering the full operational picture:

| Page | What it answers |
|---|---|
| **Delivery Performance** | OTIF rates by mode, market, and geography with date filtering |
| **Demand Trends** | Order volume, rolling averages, category seasonality, YoY growth |
| **Profitability** | Margin tiers, segment × market heatmap, top and bottom categories |
| **Shipment Mode Optimizer** | Cost vs delay tradeoffs by mode with a structured mode selection guide |
| **Risk Scoring** | Model performance metrics + live late delivery risk scoring per order |

---

## Project Structure

```
supply-chain-analytics/
├── data/
│   └── dataco_supply_chain.parquet      ← cleaned dataset, ready to use
├── models/
│   └── lgbm_late_delivery.pkl           ← trained risk classifier
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_delivery_performance.ipynb
│   ├── 03_demand_trends.ipynb
│   ├── 04_profitability_analysis.ipynb
│   └── 05_late_delivery_risk_model.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_utils.py
│   └── viz_utils.py
├── streamlit_app/
│   ├── app.py
│   └── pages/
│       ├── 1_Delivery_Performance.py
│       ├── 2_Demand_Trends.py
│       ├── 3_Profitability.py
│       ├── 4_Shipment_Mode_Optimizer.py
│       └── 5_Risk_Scoring.py
├── requirements.txt
└── README.md
```

---

## Dataset

**DataCo Smart Supply Chain for Big Data Analysis** — [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

The cleaned dataset is included as a parquet file — no additional setup needed. If you want to re-run the full pipeline from the raw CSV, download it from Kaggle and `data_loader.py` will pick it up automatically.

180,519 orders · Jan 2015 – Jan 2018 · 5 markets · 4 shipping modes

---

## How to Run Locally

```bash
conda create -n supply-chain python=3.10
conda activate supply-chain
pip install -r requirements.txt

# Launch the dashboard
cd streamlit_app
streamlit run app.py
```

Notebooks 01–05 can be run in sequence to reproduce the full analysis. Notebook 05 retrains and saves the risk model.

---

## Tools & Skills

Python · pandas · Plotly · Streamlit · LightGBM · SHAP · scikit-learn · Matplotlib · Seaborn · Supply Chain Analytics · Predictive Modelling · Data Visualisation · Business Intelligence

---

## About

**Manish Thumma** — Business Data Analyst

- LinkedIn: [linkedin.com/in/balamanishreddythumma](https://www.linkedin.com/in/balamanishreddythumma)
- GitHub: [github.com/ManishThumma](https://github.com/ManishThumma)
