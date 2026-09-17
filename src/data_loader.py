"""
data_loader.py
--------------
Handles downloading, loading, and SQLite storage of the UCI Online Retail II dataset.

Dataset  : UCI Online Retail II (2010-2011 transactions)
Source   : https://archive.ics.uci.edu/dataset/502/online+retail+ii
Licence  : CC BY 4.0  (redistribution permitted with attribution)
Size     : ~6 MB (zipped)

This module is responsible for:
  1. Downloading the dataset if not present locally.
  2. Loading the raw Excel file into a Pandas DataFrame.
  3. Persisting the cleaned DataFrame into a SQLite database.
  4. Providing a thin SQL query helper so the rest of the app can
     query the DB with plain SQL strings.
"""

import sqlite3
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── Paths (all relative to project root) ──────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent          # project root
DATA_DIR      = ROOT_DIR / "data"                     # data/
DB_PATH       = DATA_DIR / "customer_analytics.db"    # SQLite DB

DATASET_URL   = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP_FILENAME  = "online_retail_II.zip"
XLSX_FILENAME = "online_retail_II.xlsx"


# ── Download / Load ─────────────────────────────────────────────────────────────

def download_dataset(force: bool = False) -> Path:
    """
    Download the UCI Online Retail II dataset if not already on disk.

    Returns the local path to the .xlsx file.
    Raises requests.HTTPError if the download fails.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = DATA_DIR / XLSX_FILENAME

    if xlsx_path.exists() and not force:
        return xlsx_path          # already cached

    zip_path = DATA_DIR / ZIP_FILENAME
    print(f"Downloading dataset from UCI ML Repository…")

    response = requests.get(DATASET_URL, stream=True, timeout=300)
    response.raise_for_status()

    with open(zip_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=16_384):
            fh.write(chunk)

    print("Extracting…")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)

    # Remove the zip after extraction
    if zip_path.exists():
        zip_path.unlink()

    # The zip may name the file slightly differently — find the first xlsx
    if not xlsx_path.exists():
        candidates = list(DATA_DIR.glob("*.xlsx"))
        if candidates:
            candidates[0].rename(xlsx_path)
        else:
            raise FileNotFoundError(
                "Could not locate the Excel file after extraction.\n"
                "Please download it manually from:\n"
                "https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
                f"and place it at: {xlsx_path}"
            )

    return xlsx_path


def load_raw_data(filepath: Optional[Path] = None,
                  sheet: str = "Year 2010-2011") -> pd.DataFrame:
    """
    Load raw transaction data from the Excel file.

    The workbook has two sheets:
      - 'Year 2009-2010'
      - 'Year 2010-2011'   ← default (single cohort, ~500k rows)

    CustomerID is loaded as str to prevent float conversion of IDs.
    """
    if filepath is None:
        filepath = DATA_DIR / XLSX_FILENAME
        if not filepath.exists():
            filepath = download_dataset()

    df = pd.read_excel(
        filepath,
        sheet_name=sheet,
        engine="openpyxl",
        dtype={"Customer ID": str},   # prevent 12345.0 CustomerID values
    )
    return df


# ── SQLite ──────────────────────────────────────────────────────────────────────

def load_to_sqlite(df: pd.DataFrame, table: str = "transactions") -> Path:
    """
    Persist a cleaned DataFrame into the local SQLite database.

    Creates useful indexes for faster analytical queries.
    Returns the path to the database file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    df.to_sql(table, conn, if_exists="replace", index=False)

    # Indexes speed up the GROUP BY queries used in the dashboard
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_customer ON {table}(CustomerID);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_invoice  ON {table}(InvoiceNo);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_date     ON {table}(InvoiceDate);")
    conn.commit()
    conn.close()
    return DB_PATH


def get_connection() -> sqlite3.Connection:
    """Return an open SQLite connection. Caller is responsible for closing it."""
    return sqlite3.connect(DB_PATH)


def run_sql(query: str, params: tuple = ()) -> pd.DataFrame:
    """
    Execute a SQL query against the local SQLite DB and return a DataFrame.

    Example
    -------
    >>> df = run_sql(SQL_TOP_CUSTOMERS)
    """
    conn = get_connection()
    try:
        result = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return result


# ── Analytical SQL queries ──────────────────────────────────────────────────────
# These are pure SQL so they are easy to read and demonstrate SQL proficiency.

SQL_TOTAL_REVENUE = """
    SELECT ROUND(SUM(Quantity * UnitPrice), 2) AS total_revenue
    FROM transactions;
"""

SQL_REVENUE_BY_COUNTRY = """
    SELECT
        Country,
        ROUND(SUM(Quantity * UnitPrice), 2)  AS revenue,
        COUNT(DISTINCT InvoiceNo)            AS orders,
        COUNT(DISTINCT CustomerID)           AS customers
    FROM transactions
    GROUP BY Country
    ORDER BY revenue DESC;
"""

SQL_REVENUE_BY_MONTH = """
    SELECT
        strftime('%Y-%m', InvoiceDate)       AS month,
        ROUND(SUM(Quantity * UnitPrice), 2)  AS revenue,
        COUNT(DISTINCT InvoiceNo)            AS orders,
        COUNT(DISTINCT CustomerID)           AS customers
    FROM transactions
    GROUP BY month
    ORDER BY month;
"""

SQL_TOP_CUSTOMERS = """
    SELECT
        CustomerID,
        ROUND(SUM(Quantity * UnitPrice), 2)  AS total_revenue,
        COUNT(DISTINCT InvoiceNo)            AS total_orders,
        ROUND(AVG(Quantity * UnitPrice), 2)  AS avg_order_value
    FROM transactions
    GROUP BY CustomerID
    ORDER BY total_revenue DESC
    LIMIT 20;
"""

SQL_TOP_PRODUCTS = """
    SELECT
        Description,
        SUM(Quantity)                        AS total_quantity,
        ROUND(SUM(Quantity * UnitPrice), 2)  AS total_revenue,
        COUNT(DISTINCT InvoiceNo)            AS order_count
    FROM transactions
    GROUP BY Description
    ORDER BY total_revenue DESC
    LIMIT 20;
"""

SQL_ORDER_COUNTS = """
    SELECT
        COUNT(DISTINCT InvoiceNo)   AS total_orders,
        COUNT(DISTINCT CustomerID)  AS total_customers,
        COUNT(*)                    AS total_line_items
    FROM transactions;
"""

SQL_AVG_ORDER_VALUE = """
    SELECT ROUND(AVG(order_total), 2) AS avg_order_value
    FROM (
        SELECT InvoiceNo, SUM(Quantity * UnitPrice) AS order_total
        FROM transactions
        GROUP BY InvoiceNo
    ) t;
"""

SQL_CUSTOMER_PURCHASE_FREQUENCY = """
    SELECT
        CustomerID,
        COUNT(DISTINCT InvoiceNo) AS purchase_count
    FROM transactions
    GROUP BY CustomerID
    ORDER BY purchase_count DESC;
"""
