"""
rfm.py
------
Calculates Recency, Frequency, and Monetary (RFM) metrics per customer.

RFM is a classic CRM framework used in marketing analytics:
  Recency   (R) — days since the customer's most recent purchase.
                  Lower = more recent = better customer.
  Frequency (F) — number of distinct orders placed.
                  Higher = more engaged customer.
  Monetary  (M) — total revenue the customer has generated.
                  Higher = more valuable customer.

Why RFM?
  - Simple to compute from transactional data.
  - Interpretable by non-technical stakeholders.
  - Strong predictor of future customer value.
  - Standard in CRM and direct-marketing analytics.
"""

from typing import Optional

import pandas as pd


def compute_rfm(df: pd.DataFrame,
                snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary metrics for each customer.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions with columns:
        ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'TotalPrice']
    snapshot_date : pd.Timestamp, optional
        Reference date for recency calculation.
        Defaults to one day after the latest transaction in the dataset.
        Using max_date + 1 day means the most recent customer has Recency = 1.

    Returns
    -------
    pd.DataFrame with columns:
        CustomerID, Recency (int), Frequency (int), Monetary (float)
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda dates: (snapshot_date - dates.max()).days
            ),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice",  "sum"),
        )
        .reset_index()
    )

    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm


def add_rfm_scores(rfm: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Add quintile scores (1–5) for R, F, M and a composite total score.

    Scoring rules:
      R_Score : 5 = most recent, 1 = least recent  (inverted rank)
      F_Score : 5 = most frequent,   1 = least frequent
      M_Score : 5 = highest spender, 1 = lowest spender

    RFM_Total = R_Score + F_Score + M_Score  (range 3–15)

    pd.qcut is used with rank(method='first') for tie-breaking on F and M
    to avoid duplicate bin-edge errors on heavily skewed distributions.
    """
    rfm = rfm.copy()

    # Recency: reverse ranking (lower recency → higher score)
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"],
        q=n_bins,
        labels=list(range(n_bins, 0, -1)),
        duplicates="drop",
    ).astype(int)

    # Frequency: normal ranking (higher frequency → higher score)
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        q=n_bins,
        labels=list(range(1, n_bins + 1)),
        duplicates="drop",
    ).astype(int)

    # Monetary: normal ranking
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        q=n_bins,
        labels=list(range(1, n_bins + 1)),
        duplicates="drop",
    ).astype(int)

    rfm["RFM_Total"]  = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    rfm["RFM_String"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    return rfm
