"""
campaign_model.py
-----------------
Campaign-response propensity model using Logistic Regression.

⚠️  TARGET VARIABLE — DERIVED (NOT real campaign-response ground truth)
======================================================================
The UCI Online Retail II dataset contains NO campaign-response labels.
We therefore construct a PROXY target that is analytically meaningful:

    "Did the customer make at least one purchase in the final 90 days
     of the dataset, given their purchase history before that window?"

This is a *customer re-engagement propensity* model — a legitimate ML
exercise that demonstrates the same methodology used in real campaign
analytics. All results are labelled 'Analytical Demonstration Only'.

Anti-leakage design
-------------------
All features are computed from the HISTORICAL period (before the holdout
cutoff). The target is labelled from the HOLDOUT period. There is therefore
no leakage: the model cannot 'see' information from the period it is
predicting.

Pipeline
--------
1. Split: last HOLDOUT_DAYS form the holdout; everything before is history.
2. Features from history only: Recency, Frequency, Monetary, AvgOrderValue,
   UniqueProducts, AvgQuantity.
3. Target: 1 if customer purchased in holdout window, else 0.
4. Train/test split (75/25, stratified).
5. StandardScaler fit on training set only (no leakage).
6. Logistic Regression with class_weight='balanced' (handles class imbalance).
7. Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
8. Score all historical customers with propensity probabilities.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
HOLDOUT_DAYS = 90    # last N days used as the 'holdout' prediction window


# ── Feature / target construction ─────────────────────────────────────────────

def build_features_target(
    df: pd.DataFrame,
    holdout_days: int = HOLDOUT_DAYS,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Construct the feature matrix and binary target from transaction data.

    Returns
    -------
    X          : pd.DataFrame  — feature matrix (one row per customer)
    y          : pd.Series     — binary target (1 = re-purchased, 0 = did not)
    customers  : pd.Series     — matching CustomerID index
    full_df    : pd.DataFrame  — X + CustomerID + Target (for scoring later)
    """
    max_date    = df["InvoiceDate"].max()
    cutoff_date = max_date - pd.Timedelta(days=holdout_days)

    hist = df[df["InvoiceDate"] <  cutoff_date].copy()
    hold = df[df["InvoiceDate"] >= cutoff_date].copy()

    # Only model customers who have historical purchases
    # (no information available for brand-new customers)
    if hist.empty:
        raise ValueError(
            "Historical period is empty. "
            f"The dataset may have fewer than {HOLDOUT_DAYS} days of data."
        )

    snapshot = cutoff_date

    # Feature engineering on historical data ONLY
    features = (
        hist.groupby("CustomerID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x: (snapshot - x.max()).days
            ),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
            AvgOrderValue=("TotalPrice", "mean"),
            UniqueProducts=("StockCode", "nunique"),
            AvgQuantity=("Quantity", "mean"),
        )
        .reset_index()
    )
    features["Monetary"]      = features["Monetary"].round(2)
    features["AvgOrderValue"] = features["AvgOrderValue"].round(2)

    # Target: did this customer purchase in the holdout window?
    buyers_in_holdout = set(hold["CustomerID"].unique())
    features["Target"] = features["CustomerID"].isin(buyers_in_holdout).astype(int)

    X         = features.drop(columns=["CustomerID", "Target"])
    y         = features["Target"]
    customers = features["CustomerID"]

    return X, y, customers, features


# ── Model training ─────────────────────────────────────────────────────────────

def train_model(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[LogisticRegression, StandardScaler,
           np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Train Logistic Regression with proper train/test split and scaling.

    Scaler is fit ONLY on X_train to prevent data leakage into X_test.

    Returns
    -------
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,          # preserve class distribution in both splits
    )

    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)   # fit+transform on train
    X_test_s   = scaler.transform(X_test)         # transform only on test

    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight="balanced",   # compensate for class imbalance
        solver="lbfgs",
    )
    model.fit(X_train_s, y_train)

    return model, scaler, X_train_s, X_test_s, y_train, y_test


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model: LogisticRegression,
    X_test_scaled: np.ndarray,
    y_test: pd.Series,
) -> Dict:
    """
    Evaluate the trained model and return a metrics dictionary.

    Metrics chosen:
      Accuracy  : overall correctness
      Precision : of predicted positives, how many are actually positive
      Recall    : of actual positives, how many did we capture
      F1        : harmonic mean of precision and recall
      ROC-AUC   : area under the ROC curve; threshold-independent
    """
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    return {
        "accuracy":               round(float(accuracy_score(y_test, y_pred)), 4),
        "precision":              round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":                 round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":                     round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":                round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix":       confusion_matrix(y_test, y_pred).tolist(),
        "classification_report":  classification_report(y_test, y_pred),
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_all_customers(
    model: LogisticRegression,
    scaler: StandardScaler,
    X: pd.DataFrame,
    customers: pd.Series,
    rfm: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Predict re-engagement propensity for every customer in the feature set.

    Propensity is the model's predicted probability of class 1 (re-engaged).

    Priority tiers:
      High Priority   : propensity >= 0.65
      Medium Priority : 0.40 <= propensity < 0.65
      Low Priority    : propensity < 0.40
    """
    X_scaled = scaler.transform(X)
    proba    = model.predict_proba(X_scaled)[:, 1]

    scored = pd.DataFrame({
        "CustomerID": customers.values,
        "Propensity": proba.round(4),
    })

    # Merge segment info if available
    if rfm is not None and "Segment" in rfm.columns:
        scored = scored.merge(
            rfm[["CustomerID", "Segment", "Recency", "Frequency", "Monetary"]],
            on="CustomerID",
            how="left",
        )

    scored["Priority"] = pd.cut(
        scored["Propensity"],
        bins=[-0.001, 0.40, 0.65, 1.001],
        labels=["Low Priority", "Medium Priority", "High Priority"],
    )

    return scored.sort_values("Propensity", ascending=False).reset_index(drop=True)


# ── Plotly charts ─────────────────────────────────────────────────────────────

def fig_confusion_matrix(cm: List[List[int]]) -> go.Figure:
    """Annotated heatmap of the confusion matrix."""
    labels = ["Not Re-engaged (0)", "Re-engaged (1)"]
    z = cm
    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=str(val),
                    showarrow=False,
                    font=dict(color="white" if val > max(max(row) for row in z) / 2 else "black", size=16),
                )
            )
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion Matrix (Test Set)",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        annotations=annotations,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
    )
    return fig


def fig_propensity_distribution(scored: pd.DataFrame) -> go.Figure:
    """Stacked histogram of propensity scores coloured by priority tier."""
    fig = px.histogram(
        scored,
        x="Propensity",
        nbins=30,
        color="Priority",
        title="Customer Re-engagement Propensity Distribution",
        labels={"Propensity": "Propensity Score (P(re-purchase))", "count": "Customers"},
        color_discrete_map={
            "High Priority":   "#2dc653",
            "Medium Priority": "#f77f00",
            "Low Priority":    "#adb5bd",
        },
        barmode="stack",
    )
    fig.add_vline(x=0.40, line_dash="dash", line_color="orange",
                  annotation_text="Medium threshold (0.40)")
    fig.add_vline(x=0.65, line_dash="dash", line_color="green",
                  annotation_text="High threshold (0.65)")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return fig


def fig_feature_importance(
    model: LogisticRegression,
    feature_names: List[str],
) -> go.Figure:
    """Horizontal bar chart of Logistic Regression coefficients."""
    coefs = pd.DataFrame({
        "Feature":     feature_names,
        "Coefficient": model.coef_[0],
    }).sort_values("Coefficient")

    fig = px.bar(
        coefs,
        x="Coefficient",
        y="Feature",
        orientation="h",
        title="Feature Importance (Logistic Regression Coefficients)",
        labels={"Coefficient": "Coefficient (log-odds)", "Feature": ""},
        color="Coefficient",
        color_continuous_scale="RdBu",
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig


def fig_roc_curve_placeholder() -> go.Figure:
    """
    Placeholder note when the ROC curve is not available.
    (Full ROC curve requires saving y_test and y_proba, which would
    require passing them through app.py cache — kept simple here.)
    """
    fig = go.Figure()
    fig.add_annotation(
        text="ROC-AUC reported in metrics above.<br>Full curve available in the Jupyter notebook.",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14),
    )
    fig.update_layout(
        title="ROC Curve",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=250,
    )
    return fig
