import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib

SLATE_BLUE = "#4A6FA5"
CORAL = "#E8735A"
PALETTE = [SLATE_BLUE, CORAL, "#6BAB8A", "#F5B942", "#9B7FBF"]


def plot_on_time_rate_by_mode(df: pd.DataFrame) -> go.Figure:
    summary = (
        df.groupby("shipping_mode")["is_late"]
        .agg(total="count", late="sum")
        .assign(on_time=lambda x: x["total"] - x["late"])
        .assign(on_time_pct=lambda x: (x["on_time"] / x["total"] * 100).round(1))
        .reset_index()
    )

    fig = go.Figure()
    fig.add_bar(
        x=summary["shipping_mode"],
        y=summary["on_time_pct"],
        name="On Time",
        marker_color=SLATE_BLUE,
    )
    fig.add_bar(
        x=summary["shipping_mode"],
        y=(100 - summary["on_time_pct"]),
        name="Late",
        marker_color=CORAL,
    )
    fig.update_layout(
        barmode="stack",
        title="On-Time vs Late Delivery Rate by Shipping Mode",
        yaxis_title="% of Orders",
        xaxis_title="Shipping Mode",
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_monthly_order_volume(df: pd.DataFrame) -> go.Figure:
    monthly = (
        df.set_index("order_date")
        .resample("D")["order_quantity"]
        .sum()
        .reset_index()
    )
    monthly["ma30"] = monthly["order_quantity"].rolling(30).mean()
    monthly["ma90"] = monthly["order_quantity"].rolling(90).mean()

    fig = go.Figure()
    fig.add_scatter(
        x=monthly["order_date"], y=monthly["order_quantity"],
        mode="lines", name="Daily Volume",
        line=dict(color=SLATE_BLUE, width=1), opacity=0.4
    )
    fig.add_scatter(
        x=monthly["order_date"], y=monthly["ma30"],
        mode="lines", name="30-Day MA",
        line=dict(color=CORAL, width=2)
    )
    fig.add_scatter(
        x=monthly["order_date"], y=monthly["ma90"],
        mode="lines", name="90-Day MA",
        line=dict(color=PALETTE[2], width=2)
    )
    fig.update_layout(
        title="Daily Order Volume with Rolling Averages",
        yaxis_title="Order Quantity",
        xaxis_title="Date",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_profit_by_segment_region(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.groupby(["customer_segment", "market"])["order_profit"]
        .mean()
        .round(2)
        .unstack(fill_value=0)
    )

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0, CORAL], [0.5, "#FFFFFF"], [1, SLATE_BLUE]],
            text=pivot.values.round(1),
            texttemplate="%{text}",
            colorbar=dict(title="Avg Profit ($)"),
        )
    )
    fig.update_layout(
        title="Average Profit per Order: Segment × Market",
        xaxis_title="Market",
        yaxis_title="Customer Segment",
    )
    return fig


def plot_shap_summary(shap_values: np.ndarray, X_test: pd.DataFrame) -> plt.Figure:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="dot",
        show=False,
        color_bar=True,
    )
    fig = plt.gcf()
    fig.patch.set_facecolor("#0e1117")
    fig.tight_layout()
    return fig
