"""
preprocessing.py
----------------
Cleans and validates the raw UCI Online Retail II dataset.

Every cleaning decision is documented so it can be explained in an interview.

Cleaning steps applied (in order):
  1. Rename 'Customer ID' (with space) → 'CustomerID' for SQL compatibility.
  2. Drop exact duplicate rows.
  3. Separate cancellation records (InvoiceNo starts with 'C').
  4. Rename 'Invoice' → 'InvoiceNo' and 'Price' → 'UnitPrice' for clarity.
  5. Remove rows where CustomerID is missing.
  6. Remove rows where Quantity <= 0.
  7. Remove rows where UnitPrice <= 0.
  8. Parse InvoiceDate to datetime.
  9. Derive TotalPrice, YearMonth, DayOfWeek, Hour.
"""

from typing import Tuple, Dict, Any

import pandas as pd


def validate_raw(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Produce a data-quality snapshot of the raw DataFrame.

    Returns a dict of quality metrics used by the dashboard's Data Quality page.
    """
    # Determine which CustomerID column name is present
    cid_col = "Customer ID" if "Customer ID" in df.columns else "CustomerID"

    report: Dict[str, Any] = {
        "total_rows":              len(df),
        "total_columns":           len(df.columns),
        "column_names":            df.columns.tolist(),
        "dtypes":                  df.dtypes.astype(str).to_dict(),
        "missing_per_column":      df.isnull().sum().to_dict(),
        "missing_pct_per_column":  (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicate_rows":          int(df.duplicated().sum()),
        "negative_quantity_rows":  int((df["Quantity"] < 0).sum()),
        "missing_customer_id":     int(df[cid_col].isnull().sum()),
    }

    # UnitPrice or Price?
    price_col = "Price" if "Price" in df.columns else "UnitPrice"
    report["zero_or_negative_price_rows"] = int((df[price_col] <= 0).sum())

    return report


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply all cleaning steps and return (cleaned_df, cleaning_log).

    The cleaning_log is a dict that is displayed in the Data Quality page so
    users can see exactly what was removed and why.
    """
    log: Dict[str, Any] = {}
    original_rows = len(df)
    df = df.copy()

    # ── Step 1: Normalise column names ─────────────────────────────────────────
    # The 2010-2011 sheet uses 'Customer ID' (with a space).
    # Rename so the column is SQL-safe and consistent with the rest of the code.
    if "Customer ID" in df.columns:
        df = df.rename(columns={"Customer ID": "CustomerID"})

    # ── Step 2: Drop exact duplicate rows ──────────────────────────────────────
    dupes_found = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log["duplicates_removed"] = dupes_found

    # ── Step 3: Separate cancellations ─────────────────────────────────────────
    # Cancelled invoices start with 'C' (e.g. C536379).
    # These represent returns/refunds. We log the count but exclude them
    # from revenue analysis to avoid double-counting negative transactions.
    invoice_col = "Invoice" if "Invoice" in df.columns else "InvoiceNo"
    cancelled_mask = df[invoice_col].astype(str).str.startswith("C", na=False)
    log["cancellation_rows_excluded"] = int(cancelled_mask.sum())
    df = df[~cancelled_mask].copy()

    # ── Step 4: Rename columns to standardised names ───────────────────────────
    rename_map = {}
    if "Invoice" in df.columns:
        rename_map["Invoice"] = "InvoiceNo"
    if "Price" in df.columns:
        rename_map["Price"] = "UnitPrice"
    df = df.rename(columns=rename_map)

    # ── Step 5: Remove rows with missing CustomerID ────────────────────────────
    # We cannot attribute revenue to an unknown customer.
    # These are likely guest/anonymous transactions.
    missing_cid = int(df["CustomerID"].isnull().sum())
    df = df.dropna(subset=["CustomerID"])
    log["missing_customer_id_removed"] = missing_cid

    # ── Step 6: Remove rows where Quantity <= 0 ────────────────────────────────
    # After separating cancellations, remaining zero/negative quantities
    # are data-entry errors.
    invalid_qty = int((df["Quantity"] <= 0).sum())
    df = df[df["Quantity"] > 0].copy()
    log["invalid_quantity_removed"] = invalid_qty

    # ── Step 7: Remove rows where UnitPrice <= 0 ──────────────────────────────
    # Zero-price rows are often test or gift records that don't represent real
    # economic activity. Negative prices should not exist after removing
    # cancellations.
    invalid_price = int((df["UnitPrice"] <= 0).sum())
    df = df[df["UnitPrice"] > 0].copy()
    log["invalid_price_removed"] = invalid_price

    # ── Step 8: Parse InvoiceDate ──────────────────────────────────────────────
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    unparseable_dates = int(df["InvoiceDate"].isnull().sum())
    df = df.dropna(subset=["InvoiceDate"])
    log["unparseable_dates_removed"] = unparseable_dates

    # ── Step 9: Derived features ───────────────────────────────────────────────
    # TotalPrice: the actual revenue for each line item
    df["TotalPrice"] = (df["Quantity"] * df["UnitPrice"]).round(2)

    # YearMonth: used for monthly trend charts
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # DayOfWeek: used for day-of-week order pattern analysis
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

    # Hour: used for hourly revenue/order pattern analysis
    df["Hour"] = df["InvoiceDate"].dt.hour

    # Clean CustomerID: strip whitespace and remove any float artefacts
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    log["rows_removed_total"] = original_rows - len(df)
    log["rows_remaining"]     = len(df)
    log["unique_customers"]   = df["CustomerID"].nunique()
    log["unique_invoices"]    = df["InvoiceNo"].nunique()
    log["date_range"]         = f"{df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}"

    return df, log
