# Supply Chain Performance Analytics

End-to-end supply chain analytics portfolio project covering delivery performance, demand forecasting, profitability analysis, and late delivery risk prediction.

## Business Context

Late deliveries and poor margin visibility cost retailers and logistics operators millions annually in customer churn, carrier penalties, and misallocated inventory. Companies like Amazon, DHL, and Kroger invest heavily in predictive operations tooling to shift from reactive incident management to proactive fulfilment decisions. This project replicates that analytical workflow on a real-world supply chain dataset — from exploratory analysis through to a deployable risk scoring interface.

## Dataset

**DataCo Smart Supply Chain for Big Data Analysis** — available on Kaggle.

Download link: [https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

After downloading, place the file here:

```
supply-chain-analytics/
└── data/
    └── dataco_supply_chain.csv
```

The dataset covers 180,000+ orders across global markets with fields for shipping mode, delivery status, customer segment, product category, order financials, and geographic attributes.

## Project Structure

```
supply-chain-analytics/
├── data/
│   └── dataco_supply_chain.csv          # Place Kaggle dataset here
├── models/
│   └── lgbm_late_delivery.pkl           # Saved after running notebook 05
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

## How to Run

**1. Set up environment**

```bash
conda create -n supply-chain python=3.10
conda activate supply-chain
pip install -r requirements.txt
```

**2. Place the dataset**

Download from Kaggle (link above) and save to `data/dataco_supply_chain.csv`.

**3. Run notebooks in order**

```bash
cd notebooks
jupyter notebook
```

Run notebooks 01 through 05 sequentially. Notebook 05 trains and saves the risk model to `models/`.

**4. Launch the Streamlit dashboard**

```bash
cd streamlit_app
streamlit run app.py
```

## Key Findings

- **Standard Class shipping drives the majority of late deliveries** — it accounts for the highest order volume and consistently underperforms its scheduled SLA window, making it the highest-priority target for carrier renegotiation.
- **Latin America and Pacific Asia markets show structurally elevated late delivery rates** regardless of shipping mode, suggesting last-mile infrastructure constraints rather than carrier selection issues.
- **Roughly 20–25% of orders are loss-generating** at the order level, concentrated in specific product categories that are frequently shipped via premium modes — a mode-shift policy alone could recover meaningful margin.
- **The LightGBM risk classifier achieves AUC > 0.85** using only pre-shipment features, confirming that late delivery risk is largely predictable at the point of order fulfilment — before any carrier handoff occurs.
- **Scheduled delivery days is a stronger risk signal than order value** — carriers appear to treat longer lead-time commitments as lower priority, which compounds SLA risk on orders already flagged as non-urgent.

## Skills Demonstrated

- **Python** — pandas, NumPy, data wrangling at scale
- **LightGBM** — gradient boosted classifier with class imbalance handling
- **SHAP** — TreeExplainer for model interpretability and feature attribution
- **Streamlit** — multi-page interactive analytics dashboard
- **Plotly / Matplotlib / Seaborn** — publication-quality and interactive visualisations
- **Feature Engineering** — delivery delta computation, profitability tiering, label encoding
- **Predictive Modelling** — train/test split, AUC-ROC, confusion matrix, classification report
- **Supply Chain Analytics** — OTIF analysis, demand trend decomposition, shipment mode optimisation

## Author

**Manish Thumma**

- GitHub: [github.com/ManishThumma](https://github.com/ManishThumma)
- LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
