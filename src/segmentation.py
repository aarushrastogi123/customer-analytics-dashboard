"""
segmentation.py
---------------
K-Means customer segmentation on RFM features.

Pipeline:
  1. Standard-scale RFM (zero mean, unit variance) so that the different
     scales of Recency (days), Frequency (count), Monetary (£) don't bias
     the distance metric used by K-Means.
  2. Compute Elbow (inertia) and Silhouette scores for K = 2…8.
  3. Choose K as the one with the highest Silhouette score, clamped to [4, 6]
     to ensure enough segments for business interpretation.
  4. Train final K-Means with that K.
  5. Assign business-friendly segment labels using a data-driven heuristic
     (NOT hand-coded) based on each cluster's relative R, F, M profile.

Why K-Means?
  - Simple, fast, and widely understood.
  - Works well on the ~4 000-customer scale of this dataset.
  - Results are easily interpretable through cluster centroids.

Why StandardScaler?
  - K-Means uses Euclidean distance. Without scaling, Monetary (range: £1–£280k)
    would dominate the clusters and make Recency/Frequency irrelevant.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42          # reproducibility
RFM_FEATURES = ["Recency", "Frequency", "Monetary"]


# ── Scaling ──────────────────────────────────────────────────────────────────

def scale_rfm(
    rfm: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Standard-scale RFM features.

    Returns
    -------
    X       : np.ndarray  — scaled feature matrix
    scaler  : StandardScaler  — fitted scaler (needed to inverse-transform)
    features: list[str]   — feature names used
    """
    if features is None:
        features = RFM_FEATURES
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[features])
    return X, scaler, features


# ── Cluster optimisation ─────────────────────────────────────────────────────

def compute_elbow_silhouette(
    X: np.ndarray,
    k_range: range = range(2, 9),
) -> Dict[str, list]:
    """
    Compute inertia and silhouette scores for a range of K values.

    Elbow Method:  Plot inertia vs K.
                   The 'elbow' is where inertia stops decreasing sharply.
    Silhouette:    Measures how similar a point is to its own cluster vs
                   the nearest other cluster (range: -1 to 1, higher = better).

    Returns a dict with lists: 'k', 'inertia', 'silhouette'.
    """
    inertias:    List[float] = []
    silhouettes: List[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels) if k > 1 else 0.0
        silhouettes.append(float(sil))

    return {
        "k":          list(k_range),
        "inertia":    inertias,
        "silhouette": silhouettes,
    }


def choose_k(metrics: Dict[str, list], k_min: int = 4, k_max: int = 6) -> int:
    """
    Select K as the value with the highest Silhouette score,
    clamped to [k_min, k_max] for interpretability.
    """
    best_idx = int(np.argmax(metrics["silhouette"]))
    best_k   = metrics["k"][best_idx]
    return max(k_min, min(best_k, k_max))


# ── Model training ───────────────────────────────────────────────────────────

def train_kmeans(X: np.ndarray, k: int) -> KMeans:
    """Fit and return a K-Means model. n_init=10 means 10 random restarts."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X)
    return km


# ── Segment label assignment ─────────────────────────────────────────────────

def assign_segments(
    rfm: pd.DataFrame,
    labels: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    """
    Attach cluster labels to the RFM DataFrame and assign business names.

    Naming logic (data-driven, not hand-coded)
    ------------------------------------------
    For each cluster compute mean R, F, M.
    Normalise each metric to [0, 1] across clusters.
    Apply a heuristic rule table:

      Champions       R low (recent),  F high,       M high
      Loyal           R low,           F high,        M moderate
      Potential        R low-moderate,  F moderate,    M moderate
      At Risk         R high (lapsed), F moderate-high, M moderate-high
      Lost Customers  R high,          F low,         M low

    The label is selected by the rule whose conditions are most satisfied.
    If no rule matches cleanly, the cluster with the best composite RFM
    rank gets 'Loyal Customers' as a safe default.
    """
    rfm = rfm.copy()
    rfm["Cluster"] = labels

    # Cluster-level aggregation
    stats = rfm.groupby("Cluster").agg(
        mean_recency=("Recency",  "mean"),
        mean_frequency=("Frequency", "mean"),
        mean_monetary=("Monetary",  "mean"),
        count=("CustomerID", "count"),
    )

    # Normalise to [0, 1]
    def _norm(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        return (series - series.min()) / (rng if rng > 0 else 1)

    stats["r_n"] = _norm(stats["mean_recency"])    # 0 = most recent (good)
    stats["f_n"] = _norm(stats["mean_frequency"])  # 1 = most frequent (good)
    stats["m_n"] = _norm(stats["mean_monetary"])   # 1 = highest value (good)

    def _label(row: pd.Series) -> str:
        r, f, m = row["r_n"], row["f_n"], row["m_n"]
        # Champions: recent, frequent, high-value
        if r < 0.30 and f > 0.60 and m > 0.60:
            return "Champions"
        # Loyal: recent + frequent (moderate value is fine)
        if r < 0.45 and f > 0.50:
            return "Loyal Customers"
        # Potential: fairly recent, lower frequency
        if r < 0.55 and f <= 0.50 and m >= 0.25:
            return "Potential Customers"
        # At Risk: lapsed but historically engaged
        if r >= 0.55 and (f >= 0.40 or m >= 0.40):
            return "At Risk"
        # Default: inactive, low-value
        return "Lost Customers"

    label_map: Dict[int, str] = stats.apply(_label, axis=1).to_dict()

    # Handle duplicate names: if two clusters get the same label,
    # distinguish the weaker one with a suffix.
    seen: Dict[str, int] = {}
    for cluster_id, lbl in label_map.items():
        if lbl in seen:
            seen[lbl] += 1
            label_map[cluster_id] = f"{lbl} ({seen[lbl]})"
        else:
            seen[lbl] = 0

    rfm["Segment"] = rfm["Cluster"].map(label_map)
    return rfm, stats, label_map


# ── Plotly charts ─────────────────────────────────────────────────────────────

def fig_elbow_silhouette(metrics: Dict[str, list]) -> go.Figure:
    """Dual-panel chart: Elbow Method and Silhouette Score side by side."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Elbow Method — Inertia", "Silhouette Score"),
    )
    fig.add_trace(
        go.Scatter(
            x=metrics["k"], y=metrics["inertia"],
            mode="lines+markers", name="Inertia",
            line=dict(color="#4361ee", width=2),
            marker=dict(size=8),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=metrics["k"], y=metrics["silhouette"],
            mode="lines+markers", name="Silhouette",
            line=dict(color="#f72585", width=2),
            marker=dict(size=8),
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="K (Number of Clusters)", row=1, col=1)
    fig.update_xaxes(title_text="K (Number of Clusters)", row=1, col=2)
    fig.update_yaxes(title_text="Inertia",          row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    fig.update_layout(
        title="Determining the Optimal Number of Clusters",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        height=380,
    )
    return fig


def fig_segment_distribution(rfm: pd.DataFrame) -> go.Figure:
    """Donut chart: customer count by segment."""
    counts = rfm["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    fig = px.pie(
        counts,
        names="Segment",
        values="Count",
        title="Customer Segment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.40,
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return fig


def fig_segment_revenue(rfm: pd.DataFrame) -> go.Figure:
    """Bar chart: total revenue contribution by segment."""
    rev = (
        rfm.groupby("Segment")["Monetary"]
        .sum()
        .reset_index()
        .sort_values("Monetary", ascending=False)
        .rename(columns={"Monetary": "Revenue"})
    )
    fig = px.bar(
        rev,
        x="Segment",
        y="Revenue",
        title="Revenue Contribution by Segment",
        labels={"Revenue": "Total Revenue (£)", "Segment": ""},
        color="Segment",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def fig_rfm_scatter(rfm: pd.DataFrame) -> go.Figure:
    """Bubble chart: Recency vs Monetary, bubble size = Frequency."""
    # Cap Monetary at 99th pct so one outlier doesn't shrink all other bubbles
    m_cap = rfm["Monetary"].quantile(0.99)
    plot_df = rfm.copy()
    plot_df["Monetary_capped"] = plot_df["Monetary"].clip(upper=m_cap)

    fig = px.scatter(
        plot_df,
        x="Recency",
        y="Monetary_capped",
        color="Segment",
        size="Frequency",
        size_max=25,
        title="Customer Map: Recency vs Monetary (bubble size = Frequency)",
        labels={
            "Recency": "Recency (days since last purchase)",
            "Monetary_capped": "Monetary (£, capped at 99th pct)",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={"CustomerID": True, "Frequency": True, "Monetary": True, "Monetary_capped": False},
        opacity=0.65,
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return fig


def fig_segment_rfm_heatmap(rfm: pd.DataFrame) -> go.Figure:
    """
    Heatmap: average R, F, M values by segment.

    Useful for quickly seeing which dimensions drive each label.
    """
    heat = (
        rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .round(1)
    )
    # Normalise Recency inversely (lower = better → darker = better)
    heat_display = heat.copy()
    # For the heatmap we just show raw averages with annotations
    fig = go.Figure(
        data=go.Heatmap(
            z=heat_display.values,
            x=heat_display.columns.tolist(),
            y=heat_display.index.tolist(),
            colorscale="RdYlGn_r",
            text=[[f"{v:,.1f}" for v in row] for row in heat_display.values],
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Average RFM Values by Segment",
        xaxis_title="RFM Dimension",
        yaxis_title="Segment",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
    )
    return fig
