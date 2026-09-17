"""
app.py
------
Customer Analytics & Campaign Intelligence Dashboard
Streamlit entry point.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

The dataset is automatically downloaded from the UCI ML Repository
on first run (~6 MB). Subsequent runs use the local cached copy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Make src/ importable without installing as a package ──────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import (
    DB_PATH,
    SQL_AVG_ORDER_VALUE,
    SQL_CUSTOMER_PURCHASE_FREQUENCY,
    SQL_ORDER_COUNTS,
    SQL_REVENUE_BY_COUNTRY,
    SQL_REVENUE_BY_MONTH,
    SQL_TOP_CUSTOMERS,
    SQL_TOP_PRODUCTS,
    SQL_TOTAL_REVENUE,
    download_dataset,
    load_raw_data,
    load_to_sqlite,
    run_sql,
)
from preprocessing import clean_data, validate_raw
from eda import (
    fig_customer_purchase_frequency,
    fig_daily_orders,
    fig_monthly_customers,
    fig_monthly_revenue,
    fig_order_value_distribution,
    fig_revenue_by_hour,
    fig_top_countries,
    fig_top_products,
)
from rfm import add_rfm_scores, compute_rfm
from segmentation import (
    assign_segments,
    choose_k,
    compute_elbow_silhouette,
    fig_elbow_silhouette,
    fig_rfm_scatter,
    fig_segment_distribution,
    fig_segment_revenue,
    fig_segment_rfm_heatmap,
    scale_rfm,
    train_kmeans,
)
from campaign_model import (
    HOLDOUT_DAYS,
    build_features_target,
    evaluate_model,
    fig_confusion_matrix,
    fig_feature_importance,
    fig_propensity_distribution,
    score_all_customers,
    train_model,
)
from insights import generate_insights

# ══════════════════════════════════════════════════════════════════════════════
# Streamlit page configuration
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS for KPI cards and insight cards ───────────────────────────────
st.markdown("""
<style>
/* KPI cards */
.kpi-blue   { background: linear-gradient(135deg,#4361ee,#7209b7); }
.kpi-green  { background: linear-gradient(135deg,#11998e,#38ef7d); }
.kpi-orange { background: linear-gradient(135deg,#f7971e,#ffd200); }
.kpi-red    { background: linear-gradient(135deg,#cb2d3e,#ef473a); }

.kpi-card {
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    color: #fff;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,.12);
    margin-bottom: 0.5rem;
}
.kpi-value { font-size: 2rem; font-weight: 700; margin: 0; }
.kpi-label { font-size: 0.82rem; opacity: .88; margin: 0; }

/* Insight cards */
.ins-card {
    border-left: 5px solid #4361ee;
    background: #f0f4ff;
    padding: .9rem 1.1rem;
    border-radius: 0 10px 10px 0;
    margin-bottom: .85rem;
}
.ins-warning { border-left-color:#f77f00; background:#fff8f0; }
.ins-success { border-left-color:#2dc653; background:#f0fff4; }
.ins-info    { border-left-color:#4361ee; background:#f0f4ff; }

/* Tighten the page top */
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached data pipeline — runs once per session
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_pipeline():
    """
    Execute the full data science pipeline and return all artefacts.

    Cached by Streamlit so the expensive computation only runs once.
    The cache is invalidated when the source code changes.
    """
    # 1 ─ Download & load
    xlsx = download_dataset()
    raw  = load_raw_data(xlsx)

    # 2 ─ Validate & clean
    quality = validate_raw(raw)
    df, cleaning_log = clean_data(raw)

    # 3 ─ SQLite (so the SQL section actually works against a real DB)
    load_to_sqlite(df)

    # 4 ─ RFM
    rfm_base   = compute_rfm(df)
    rfm_scored = add_rfm_scores(rfm_base)

    # 5 ─ Segmentation
    X, scaler_km, features = scale_rfm(rfm_base)
    elbow_metrics = compute_elbow_silhouette(X, k_range=range(2, 9))
    best_k        = choose_k(elbow_metrics, k_min=4, k_max=6)
    km_model      = train_kmeans(X, best_k)
    rfm_seg, cluster_stats, label_map = assign_segments(rfm_base, km_model.labels_)

    # 6 ─ Campaign propensity model
    X_feat, y, customers, feat_df = build_features_target(df)
    model, model_scaler, X_tr, X_te, y_tr, y_te = train_model(X_feat, y)
    eval_metrics = evaluate_model(model, X_te, y_te)
    scored       = score_all_customers(model, model_scaler, X_feat, customers, rfm_seg)

    # 7 ─ Business insights
    insights = generate_insights(df, rfm_seg, scored)

    return {
        "raw":            raw,
        "df":             df,
        "quality":        quality,
        "cleaning_log":   cleaning_log,
        "rfm":            rfm_seg,
        "rfm_scored":     rfm_scored,
        "cluster_stats":  cluster_stats,
        "label_map":      label_map,
        "elbow_metrics":  elbow_metrics,
        "best_k":         best_k,
        "model":          model,
        "model_scaler":   model_scaler,
        "X_feat":         X_feat,
        "y":              y,
        "eval_metrics":   eval_metrics,
        "scored":         scored,
        "feat_df":        feat_df,
        "insights":       insights,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Customer Analytics")
    st.markdown("*Campaign Intelligence Dashboard*")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        options=[
            "📊 Executive Dashboard",
            "🔍 Data Quality & EDA",
            "👥 RFM Analytics",
            "🧩 Segmentation",
            "🎯 Campaign Intelligence",
            "💡 Business Insights",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "**Dataset:** UCI Online Retail II  \n"
        "**Licence:** CC BY 4.0  \n"
        "**Source:** archive.ics.uci.edu"
    )

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading pipeline… (first run downloads ~6 MB dataset)"):
    try:
        data = run_pipeline()
    except Exception as exc:
        st.error(
            f"**Pipeline error:** {exc}\n\n"
            "If the dataset failed to download automatically, please:\n"
            "1. Visit https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
            "2. Download the zip and extract `online_retail_II.xlsx` into the `data/` folder.\n"
            "3. Restart the app."
        )
        st.stop()

# Unpack for convenience
df           = data["df"]
rfm          = data["rfm"]
eval_metrics = data["eval_metrics"]
scored       = data["scored"]
insights     = data["insights"]


# ══════════════════════════════════════════════════════════════════════════════
# Helper: KPI card HTML
# ══════════════════════════════════════════════════════════════════════════════
def kpi(label: str, value: str, colour: str = "blue") -> str:
    return (
        f'<div class="kpi-card kpi-{colour}">'
        f'<p class="kpi-label">{label}</p>'
        f'<p class="kpi-value">{value}</p>'
        f"</div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":
    st.title("📊 Executive Dashboard")
    st.markdown(
        "High-level KPIs and trends. "
        "All figures come from SQL queries executed against the cleaned SQLite database."
    )

    # ── KPIs via SQL ──────────────────────────────────────────────────────────
    total_rev    = run_sql(SQL_TOTAL_REVENUE)["total_revenue"].iloc[0]
    order_stats  = run_sql(SQL_ORDER_COUNTS)
    avg_ov       = run_sql(SQL_AVG_ORDER_VALUE)["avg_order_value"].iloc[0]
    total_orders = int(order_stats["total_orders"].iloc[0])
    total_cust   = int(order_stats["total_customers"].iloc[0])

    at_risk_n    = int((rfm["Segment"] == "At Risk").sum())
    champion_n   = int((rfm["Segment"] == "Champions").sum())
    avg_rev_cust = round(total_rev / total_cust, 2) if total_cust else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Total Revenue",    f"£{total_rev:,.0f}",  "blue"),   unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Total Orders",     f"{total_orders:,}",   "green"),  unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Total Customers",  f"{total_cust:,}",     "blue"),   unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Avg Order Value",  f"£{avg_ov:,.2f}",     "orange"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    with c5:
        st.markdown(kpi("Avg Revenue / Customer", f"£{avg_rev_cust:,.2f}", "green"),  unsafe_allow_html=True)
    with c6:
        st.markdown(kpi("At-Risk Customers",       f"{at_risk_n:,}",       "red"),    unsafe_allow_html=True)
    with c7:
        st.markdown(kpi("Champion Customers",      f"{champion_n:,}",      "blue"),   unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_monthly_revenue(df), use_container_width=True)
    with col2:
        st.plotly_chart(fig_segment_distribution(rfm), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_segment_revenue(rfm), use_container_width=True)
    with col4:
        rev_month = run_sql(SQL_REVENUE_BY_MONTH)
        fig = px.bar(
            rev_month, x="month", y="orders",
            title="Monthly Order Count (SQL)",
            labels={"month": "Month", "orders": "Orders"},
            color_discrete_sequence=["#4361ee"],
        )
        fig.update_layout(xaxis_tickangle=-45, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA QUALITY & EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Data Quality & EDA":
    st.title("🔍 Data Quality & EDA")

    # ── Data quality ──────────────────────────────────────────────────────────
    st.header("1 · Data Quality Report")
    quality  = data["quality"]
    clog     = data["cleaning_log"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Rows",        f"{quality['total_rows']:,}")
    c2.metric("Columns",         quality["total_columns"])
    c3.metric("Duplicate Rows",  f"{quality['duplicate_rows']:,}")
    c4.metric("Rows Remaining",  f"{clog['rows_remaining']:,}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Missing Values (Raw Data)")
        mv = pd.DataFrame({
            "Column":        list(quality["missing_per_column"].keys()),
            "Missing Count": list(quality["missing_per_column"].values()),
            "Missing %":     list(quality["missing_pct_per_column"].values()),
        })
        st.dataframe(mv, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Data Types")
        dt = pd.DataFrame({
            "Column": list(quality["dtypes"].keys()),
            "Type":   list(quality["dtypes"].values()),
        })
        st.dataframe(dt, use_container_width=True, hide_index=True)

    st.subheader("Cleaning Log")
    st.info(
        f"| Step | Rows Affected |\n"
        f"|------|---------------|\n"
        f"| Duplicates removed | {clog.get('duplicates_removed', 0):,} |\n"
        f"| Cancellations excluded | {clog.get('cancellation_rows_excluded', 0):,} |\n"
        f"| Missing CustomerID removed | {clog.get('missing_customer_id_removed', 0):,} |\n"
        f"| Invalid Quantity removed | {clog.get('invalid_quantity_removed', 0):,} |\n"
        f"| Invalid Price removed | {clog.get('invalid_price_removed', 0):,} |\n"
        f"| Rows remaining | **{clog.get('rows_remaining', 0):,}** |\n"
        f"| Date range | {clog.get('date_range', 'N/A')} |"
    )

    st.subheader("Cleaned Dataset — Descriptive Statistics")
    st.dataframe(
        df[["Quantity", "UnitPrice", "TotalPrice"]].describe().round(2),
        use_container_width=True,
    )

    with st.expander("🔎 View first 100 rows of cleaned data"):
        st.dataframe(df.head(100), use_container_width=True)

    # ── EDA ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("2 · Exploratory Data Analysis")

    countries = ["All Countries"] + sorted(df["Country"].dropna().unique().tolist())
    sel_country = st.selectbox("Filter charts by country", countries, index=0)
    df_f = df if sel_country == "All Countries" else df[df["Country"] == sel_country]

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_monthly_revenue(df_f), use_container_width=True)
    with col2:
        st.plotly_chart(fig_daily_orders(df_f), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_top_products(df_f), use_container_width=True)
    with col4:
        st.plotly_chart(fig_top_countries(df_f), use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(fig_order_value_distribution(df_f), use_container_width=True)
    with col6:
        st.plotly_chart(fig_revenue_by_hour(df_f), use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(fig_monthly_customers(df_f), use_container_width=True)
    with col8:
        st.plotly_chart(fig_customer_purchase_frequency(df_f), use_container_width=True)

    st.subheader("Revenue by Country — SQL Query")
    with st.expander("View SQL"):
        st.code(SQL_REVENUE_BY_COUNTRY, language="sql")
    st.dataframe(run_sql(SQL_REVENUE_BY_COUNTRY).head(20), use_container_width=True, hide_index=True)

    st.subheader("Top Products — SQL Query")
    with st.expander("View SQL"):
        st.code(SQL_TOP_PRODUCTS, language="sql")
    st.dataframe(run_sql(SQL_TOP_PRODUCTS), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RFM ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 RFM Analytics":
    st.title("👥 RFM Customer Analytics")
    st.markdown("""
    **RFM** (Recency · Frequency · Monetary) is a behavioural segmentation framework
    that quantifies each customer's engagement and value from transactional history.

    | Metric | Definition | Lower is… |
    |--------|-----------|-----------|
    | **Recency** | Days since last purchase | Better (more recent) |
    | **Frequency** | Number of distinct orders | Worse (less engaged) |
    | **Monetary** | Total revenue generated | Worse (lower value) |
    """)

    st.subheader("RFM Summary Statistics")
    st.dataframe(
        rfm[["Recency", "Frequency", "Monetary"]].describe().round(2),
        use_container_width=True,
    )

    st.subheader("RFM Distributions")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.histogram(rfm, x="Recency", nbins=40, title="Recency Distribution",
                           color_discrete_sequence=["#4361ee"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        freq_cap = rfm["Frequency"].quantile(0.99)
        fig = px.histogram(rfm[rfm["Frequency"] <= freq_cap], x="Frequency",
                           nbins=40, title="Frequency Distribution",
                           color_discrete_sequence=["#f72585"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        mon_cap = rfm["Monetary"].quantile(0.99)
        fig = px.histogram(rfm[rfm["Monetary"] <= mon_cap], x="Monetary",
                           nbins=40, title="Monetary Distribution",
                           color_discrete_sequence=["#7209b7"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        fig = px.scatter(rfm, x="Recency", y="Monetary", color="Segment",
                         title="Recency vs Monetary by Segment",
                         opacity=0.55,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c5:
        fig = px.scatter(rfm, x="Frequency", y="Monetary", color="Segment",
                         title="Frequency vs Monetary by Segment",
                         opacity=0.55,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("RFM Customer Table")
    seg_opts = sorted(rfm["Segment"].dropna().unique().tolist())
    sel_segs = st.multiselect("Filter by Segment", seg_opts, default=seg_opts)
    rfm_disp = (
        rfm[rfm["Segment"].isin(sel_segs)]
        [["CustomerID", "Recency", "Frequency", "Monetary", "Segment"]]
        .sort_values("Monetary", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(rfm_disp, use_container_width=True, hide_index=True)

    st.subheader("Top Customers by Revenue — SQL")
    with st.expander("View SQL"):
        st.code(SQL_TOP_CUSTOMERS, language="sql")
    st.dataframe(run_sql(SQL_TOP_CUSTOMERS), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧩 Segmentation":
    st.title("🧩 K-Means Customer Segmentation")
    best_k = data["best_k"]
    st.markdown(f"""
    **Algorithm:** K-Means clustering on scaled RFM features.  
    **Optimal K = {best_k}**, selected by highest Silhouette Score (range 2–8),
    then clamped to [4, 6] for meaningful business interpretation.
    """)

    # K selection charts
    st.subheader("Cluster Count Optimisation")
    st.plotly_chart(fig_elbow_silhouette(data["elbow_metrics"]), use_container_width=True)

    with st.expander("📖 How was K chosen? (Interview-ready explanation)"):
        sil = dict(zip(data["elbow_metrics"]["k"], data["elbow_metrics"]["silhouette"]))
        st.markdown(f"""
        **Elbow Method:** We plot inertia (sum of squared distances to centroids)
        for K = 2…8. We look for the 'elbow' — the point where adding more clusters
        gives diminishing returns. This suggests a range of candidate K values.

        **Silhouette Score:** For each K we compute the average silhouette coefficient,
        which measures how similar each point is to its own cluster vs the nearest
        other cluster. A score of 1 is perfect; 0 means clusters overlap.

        Silhouette scores: `{sil}`

        The K with the highest silhouette within [4, 6] was chosen: **K = {best_k}**.
        We clamp to a minimum of 4 to ensure enough segments for business use.
        """)

    st.markdown("---")
    st.subheader("Cluster Characteristics")
    cluster_summary = (
        rfm.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency",   "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary",  "mean"),
            Total_Revenue=("Monetary",  "sum"),
        )
        .round(2)
        .reset_index()
    )
    cluster_summary["Revenue_Share_%"] = (
        cluster_summary["Total_Revenue"] / cluster_summary["Total_Revenue"].sum() * 100
    ).round(1)
    st.dataframe(
        cluster_summary.sort_values("Total_Revenue", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_segment_distribution(rfm), use_container_width=True)
    with c2:
        st.plotly_chart(fig_segment_revenue(rfm), use_container_width=True)

    st.plotly_chart(fig_rfm_scatter(rfm), use_container_width=True)
    st.plotly_chart(fig_segment_rfm_heatmap(rfm), use_container_width=True)

    with st.expander("📖 How were segment labels assigned? (Interview-ready)"):
        st.markdown("""
        Labels are assigned **algorithmically**, not hand-coded.

        For each cluster we compute **mean Recency, Frequency, Monetary**.
        We then normalise each dimension to [0, 1] across clusters and apply a
        heuristic rule table:

        | Label              | Recency (low=recent) | Frequency | Monetary |
        |--------------------|---------------------|-----------|----------|
        | Champions          | < 0.30              | > 0.60    | > 0.60   |
        | Loyal Customers    | < 0.45              | > 0.50    | any      |
        | Potential Customers| < 0.55              | ≤ 0.50    | ≥ 0.25   |
        | At Risk            | ≥ 0.55              | ≥ 0.40 or M ≥ 0.40 | — |
        | Lost Customers     | default             |           |          |

        The actual cluster-level stats determine which rule fires.
        If multiple clusters match the same label, a suffix is added to distinguish them.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CAMPAIGN INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Campaign Intelligence":
    st.title("🎯 Campaign Intelligence")

    st.warning(
        f"⚠️ **Derived Analytical Target — Not Real Campaign Data**\n\n"
        f"The UCI Online Retail II dataset does not contain campaign-response labels. "
        f"The model below predicts whether a customer **re-purchased within the final "
        f"{HOLDOUT_DAYS} days** of the dataset, using only their prior history as features. "
        f"This demonstrates propensity modelling methodology. "
        f"It must NOT be interpreted as real campaign-response prediction."
    )

    st.subheader("Model Configuration")
    col_cfg = st.columns(3)
    col_cfg[0].info(f"**Algorithm:** Logistic Regression")
    col_cfg[1].info(f"**Holdout Window:** {HOLDOUT_DAYS} days")
    col_cfg[2].info(f"**Class Weight:** Balanced")

    with st.expander("Features used"):
        st.markdown("""
        All features are computed from the **historical period only** (before the holdout
        cutoff) to prevent data leakage:

        | Feature | Description |
        |---------|-------------|
        | `Recency` | Days since last purchase in the historical period |
        | `Frequency` | Number of unique orders |
        | `Monetary` | Total revenue generated |
        | `AvgOrderValue` | Mean revenue per order |
        | `UniqueProducts` | Count of distinct products purchased |
        | `AvgQuantity` | Mean quantity per line item |
        """)

    # ── Evaluation metrics ────────────────────────────────────────────────────
    st.subheader("Model Evaluation (Test Set — 25% holdout)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  eval_metrics["accuracy"])
    c2.metric("Precision", eval_metrics["precision"])
    c3.metric("Recall",    eval_metrics["recall"])
    c4.metric("F1 Score",  eval_metrics["f1"])
    c5.metric("ROC-AUC",   eval_metrics["roc_auc"])

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.plotly_chart(
            fig_confusion_matrix(eval_metrics["confusion_matrix"]),
            use_container_width=True,
        )
    with col_m2:
        st.plotly_chart(
            fig_feature_importance(data["model"], list(data["X_feat"].columns)),
            use_container_width=True,
        )

    with st.expander("Full Classification Report"):
        st.text(eval_metrics["classification_report"])

    # ── Propensity scores ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Customer Propensity Scores")
    st.plotly_chart(fig_propensity_distribution(scored), use_container_width=True)

    high_n  = int((scored["Priority"] == "High Priority").sum())
    med_n   = int((scored["Priority"] == "Medium Priority").sum())
    low_n   = int((scored["Priority"] == "Low Priority").sum())

    ca, cb, cc = st.columns(3)
    ca.metric("High Priority  (≥ 0.65)", high_n)
    cb.metric("Medium Priority (0.40–0.65)", med_n)
    cc.metric("Low Priority  (< 0.40)", low_n)

    prio_opts = ["High Priority", "Medium Priority", "Low Priority"]
    sel_prio  = st.multiselect("Filter by Priority", prio_opts, default=["High Priority", "Medium Priority"])
    show_cols = ["CustomerID", "Propensity", "Priority"]
    if "Segment" in scored.columns:
        show_cols.insert(2, "Segment")
    scored_disp = (
        scored[scored["Priority"].isin(sel_prio)][show_cols]
        .reset_index(drop=True)
    )
    st.dataframe(scored_disp, use_container_width=True, hide_index=True)

    with st.expander("📖 Why Logistic Regression? (Interview-ready)"):
        st.markdown("""
        1. **Interpretability** — coefficients directly show each feature's direction and
           magnitude of effect; easy to explain to a stakeholder.
        2. **Probability output** — gives a propensity *score* (0–1), not just a class label,
           which is exactly what campaign analytics needs.
        3. **Appropriate for binary classification** — our target is binary (re-purchased or not).
        4. **Baseline** — before trying complex models, always establish a solid linear baseline.
        5. **No data leakage risk** — the StandardScaler is fit only on training data,
           then applied to test data without re-fitting.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Business Insights":
    st.title("💡 Business Insights & Recommendations")
    st.markdown(
        "All insights below are generated dynamically from the actual analytical results. "
        "No numbers are hardcoded."
    )

    for ins in insights:
        sev = ins.get("severity", "info")
        st.markdown(
            f'<div class="ins-card ins-{sev}">'
            f'<strong>📌 {ins["title"]}</strong><br>'
            f'<em>{ins["observation"]}</em><br><br>'
            f'<strong>Recommendation:</strong> {ins["recommendation"]}'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("Segment Summary Table")
    seg_tbl = (
        rfm.groupby("Segment")
        .agg(
            Customers=("CustomerID",  "count"),
            Total_Revenue=("Monetary", "sum"),
            Avg_Recency=("Recency",   "mean"),
            Avg_Frequency=("Frequency","mean"),
            Avg_Monetary=("Monetary",  "mean"),
        )
        .round(2)
        .reset_index()
    )
    seg_tbl["Revenue_Share_%"] = (
        seg_tbl["Total_Revenue"] / seg_tbl["Total_Revenue"].sum() * 100
    ).round(1)
    st.dataframe(
        seg_tbl.sort_values("Total_Revenue", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Top 10 Customers by Revenue")
    top10 = (
        rfm.nlargest(10, "Monetary")
        [["CustomerID", "Segment", "Recency", "Frequency", "Monetary"]]
        .reset_index(drop=True)
    )
    st.dataframe(top10, use_container_width=True, hide_index=True)

    st.subheader("Revenue by Month — SQL")
    with st.expander("View SQL"):
        st.code(SQL_REVENUE_BY_MONTH, language="sql")
    st.dataframe(run_sql(SQL_REVENUE_BY_MONTH), use_container_width=True, hide_index=True)
