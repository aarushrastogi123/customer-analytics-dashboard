"""
eda.py
------
Plotly visualisation functions for the EDA section of the dashboard.

Each function:
  - Accepts a cleaned Pandas DataFrame.
  - Returns a Plotly Figure (go.Figure).
  - Can be rendered with st.plotly_chart(fig, use_container_width=True).

Design decisions:
  - White backgrounds so charts look clean inside Streamlit.
  - Consistent colour palette across the dashboard.
  - Outlier capping (99th percentile) on skewed distributions to keep
    histograms readable without removing data from the main dataset.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ordered days for the day-of-week chart
_DAYS_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]


def fig_monthly_revenue(df: pd.DataFrame) -> go.Figure:
    """Line chart: total revenue aggregated by calendar month."""
    monthly = (
        df.groupby("YearMonth")["TotalPrice"]
        .sum()
        .reset_index()
        .sort_values("YearMonth")
        .rename(columns={"YearMonth": "Month", "TotalPrice": "Revenue"})
    )
    fig = px.line(
        monthly,
        x="Month",
        y="Revenue",
        title="Monthly Revenue Trend",
        markers=True,
        labels={"Revenue": "Revenue (£)", "Month": "Month"},
        color_discrete_sequence=["#4361ee"],
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def fig_daily_orders(df: pd.DataFrame) -> go.Figure:
    """Bar chart: number of unique orders by day of week."""
    daily = (
        df.groupby("DayOfWeek")["InvoiceNo"]
        .nunique()
        .reindex(_DAYS_ORDER, fill_value=0)
        .reset_index()
        .rename(columns={"DayOfWeek": "Day", "InvoiceNo": "Orders"})
    )
    fig = px.bar(
        daily,
        x="Day",
        y="Orders",
        title="Orders by Day of Week",
        color="Orders",
        color_continuous_scale="Blues",
        labels={"Orders": "Number of Orders", "Day": ""},
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig


def fig_top_products(df: pd.DataFrame, n: int = 15) -> go.Figure:
    """Horizontal bar chart: top N products by total revenue."""
    top = (
        df.groupby("Description")["TotalPrice"]
        .sum()
        .nlargest(n)
        .reset_index()
        .sort_values("TotalPrice")
        .rename(columns={"Description": "Product", "TotalPrice": "Revenue"})
    )
    fig = px.bar(
        top,
        x="Revenue",
        y="Product",
        orientation="h",
        title=f"Top {n} Products by Revenue",
        labels={"Revenue": "Revenue (£)", "Product": ""},
        color="Revenue",
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig


def fig_top_countries(df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Bar chart: top N countries by total revenue."""
    top = (
        df.groupby("Country")["TotalPrice"]
        .sum()
        .nlargest(n)
        .reset_index()
        .rename(columns={"TotalPrice": "Revenue"})
    )
    fig = px.bar(
        top,
        x="Country",
        y="Revenue",
        title=f"Top {n} Countries by Revenue",
        labels={"Revenue": "Revenue (£)", "Country": "Country"},
        color="Revenue",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig


def fig_order_value_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram: distribution of order-level total values.

    Capped at the 99th percentile to prevent a small number of very large
    B2B orders from making the chart unreadable. The cap is stated in the title.
    """
    order_vals = (
        df.groupby("InvoiceNo")["TotalPrice"]
        .sum()
        .reset_index()
        .rename(columns={"TotalPrice": "OrderValue"})
    )
    cap = order_vals["OrderValue"].quantile(0.99)
    capped = order_vals[order_vals["OrderValue"] <= cap]

    fig = px.histogram(
        capped,
        x="OrderValue",
        nbins=50,
        title=f"Order Value Distribution (capped at 99th pct ≈ £{cap:,.0f})",
        labels={"OrderValue": "Order Value (£)", "count": "Number of Orders"},
        color_discrete_sequence=["#4361ee"],
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return fig


def fig_customer_purchase_frequency(df: pd.DataFrame) -> go.Figure:
    """Histogram: number of orders per customer (capped at 99th percentile)."""
    freq = (
        df.groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index()
        .rename(columns={"InvoiceNo": "PurchaseCount"})
    )
    cap = freq["PurchaseCount"].quantile(0.99)
    capped = freq[freq["PurchaseCount"] <= cap]

    fig = px.histogram(
        capped,
        x="PurchaseCount",
        nbins=40,
        title=f"Customer Purchase Frequency (capped at 99th pct ≈ {cap:.0f} orders)",
        labels={"PurchaseCount": "Number of Orders", "count": "Number of Customers"},
        color_discrete_sequence=["#7209b7"],
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return fig


def fig_revenue_by_hour(df: pd.DataFrame) -> go.Figure:
    """Bar chart: total revenue by hour of day."""
    hourly = (
        df.groupby("Hour")["TotalPrice"]
        .sum()
        .reset_index()
        .rename(columns={"TotalPrice": "Revenue"})
    )
    fig = px.bar(
        hourly,
        x="Hour",
        y="Revenue",
        title="Revenue by Hour of Day",
        labels={"Revenue": "Revenue (£)", "Hour": "Hour (24 h)"},
        color="Revenue",
        color_continuous_scale="Oranges",
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig


def fig_monthly_customers(df: pd.DataFrame) -> go.Figure:
    """Bar chart: unique active customers per month."""
    mc = (
        df.groupby("YearMonth")["CustomerID"]
        .nunique()
        .reset_index()
        .sort_values("YearMonth")
        .rename(columns={"YearMonth": "Month", "CustomerID": "Customers"})
    )
    fig = px.bar(
        mc,
        x="Month",
        y="Customers",
        title="Active Customers per Month",
        labels={"Customers": "Unique Customers", "Month": "Month"},
        color_discrete_sequence=["#f72585"],
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
