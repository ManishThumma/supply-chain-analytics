# Supply Chain Performance Analytics

I built this project to get hands-on with the kind of operational analytics work that supply chain and logistics teams at companies like Amazon, FedEx, DHL, and Kroger do every day. The DataCo Smart Supply Chain dataset gave me 180,519 real orders across global markets spanning 2015–2018 — enough to do meaningful analysis on delivery performance, demand patterns, profitability, and late delivery prediction.

---

## What I Was Trying to Answer

A few questions drove this project:

- Where are deliveries failing, and which shipping modes and markets have the worst on-time performance?
- What does profitability actually look like at the order level — and which categories are quietly losing money?
- Can I build a model that flags high late-delivery-risk orders *before* they ship, using only information available at the time of fulfilment?

---

## Dataset

**DataCo Smart Supply Chain for Big Data Analysis** — Kaggle

[Download here](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

Place the downloaded CSV at:

```
supply-chain-analytics/data/dataco_supply_chain.csv
```

**Dataset at a glance:**
- 180,519 orders
- Jan 2015 – Jan 2018
- Markets: Europe, LATAM, Pacific Asia, USCA, Africa
- Shipping modes: Standard Class, Second Class, First Class, Same Day
- 53 columns covering order financials, shipping details, customer geography, and product categories

---

## Project Structure

```
supply-chain-analytics/
├── data/
│   └── dataco_supply_chain.csv
├── models/
│   └── lgbm_late_delivery.pkl        ← trained after running notebook 05
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

## What I Found

### Delivery Performance
- **54.8% of orders were flagged as late** — higher than I expected going in. That's not a carrier problem, that's a structural SLA misalignment.
- Standard Class had the lowest late rate at **38.1%** despite being the highest-volume mode. First Class and Second Class underperformed badly relative to their premium positioning in the shipping hierarchy — a clear sign the SLA promises don't match network capacity.
- Late delivery rates were remarkably consistent across all five markets (54–55%), which tells me this isn't a geography problem. It's an across-the-board operational issue.
- California, Puerto Rico, and Illinois led in absolute late order volume, though PR dominates because of overall order concentration there.

### Profitability
- Average profit per order came out to **$21.97**, but the distribution tells a messier story.
- **19.4% of orders were loss-generating** and 3.8% were breakeven — meaning roughly 1 in 4 orders either lost money or barely covered costs.
- **Computers** were by far the most profitable category at $157.59 average profit per order. Garden, Crafts, Cameras, and Fishing followed.
- **CDs, Toys, and Books** sat at the bottom — $1.42, $1.70, and $2.18 average profit respectively. These categories need a hard look at whether they're pulling their weight or just adding fulfilment overhead.
- Profit differences across Customer Segments (Consumer, Corporate, Home Office) were minimal — the real margin variation is driven by category and shipping mode, not who the customer is.

### Late Delivery Risk Model
- Built a **LightGBM classifier** using only pre-shipment features — no actual shipping delay included, since that's a leaker.
- Features used: shipping mode, market, customer segment, product category, order quantity, order value, and scheduled shipping days.
- Achieved an **AUC of 0.73** on the held-out test set. Not perfect, but meaningful — this model can flag high-risk orders before they leave the warehouse, which is the whole point.
- SHAP analysis showed that **scheduled shipping days and shipping mode** are the two strongest predictors. Orders with longer scheduled windows tend to slip more, and certain modes structurally underdeliver their SLA promises.

---

## How to Run

**1. Set up environment**

```bash
conda create -n supply-chain python=3.10
conda activate supply-chain
pip install -r requirements.txt
```

**2. Add the dataset**

Download from Kaggle and place at `data/dataco_supply_chain.csv`.

**3. Run notebooks in order**

```bash
jupyter notebook
```

Run 01 through 05 in sequence. Notebook 05 trains and saves the model.

**4. Launch the dashboard**

```bash
cd streamlit_app
streamlit run app.py
```

---

## Tech Stack

| Area | Tools |
|---|---|
| Data wrangling | Python, pandas, NumPy |
| Modelling | LightGBM, scikit-learn |
| Interpretability | SHAP (TreeExplainer) |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Feature engineering | Custom pipeline (delivery delta, profit tiers, label encoding) |

---

## About Me

**Manish Thumma** — Data Analyst

- GitHub: [github.com/ManishThumma](https://github.com/ManishThumma)
- LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
