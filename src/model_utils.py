import joblib
import logging
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

logger = logging.getLogger(__name__)

FEATURES = [
    "shipping_mode_enc",
    "market_enc",
    "customer_segment_enc",
    "category_name_enc",
    "order_quantity",
    "sales",
    "days_shipping_scheduled",
    "shipping_mode_market",
    "order_month",
    "order_dayofweek",
    "is_holiday_period",
    "order_value_bin",
    "scheduled_x_value",
]

MODEL_PATH = Path("models/lgbm_late_delivery.pkl")


def train_risk_model(df: pd.DataFrame):
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing encoded features: {missing}. Run encode_categoricals first.")

    X = df[FEATURES].copy()
    y = df["late_delivery_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=40,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    logger.info("AUC-ROC: %.4f", auc)
    logger.info("\n%s", classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_test)
    # LightGBM binary returns either a list [neg, pos] or a single 2D array
    if isinstance(raw, list):
        shap_values = raw[1]
    else:
        shap_values = raw

    return model, X_test, y_test, shap_values


def load_risk_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_risk_model first."
        )
    return joblib.load(MODEL_PATH)


def get_model_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "auc": roc_auc_score(y_test, y_prob),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_test,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def predict_single_order(model, order_params: dict) -> float:
    row = pd.DataFrame([order_params])[FEATURES]
    return float(model.predict_proba(row)[0, 1])
