# Customer Analytics & Campaign Intelligence Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75.svg)](https://plotly.com/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57.svg)](https://www.sqlite.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end e-commerce customer analytics and machine learning application that transforms raw transaction data into strategic customer segments and campaign propensity scores. Built with a modular Python backend, an embedded SQLite analytical database, and an interactive Streamlit dashboard.

---

## 🎯 Project Overview & Business Value

Retail businesses frequently struggle with customer retention, untargeted marketing spend, and high churn rates. This project demonstrates how data science bridges the gap between raw transactional databases and actionable marketing operations:

1. **Data Auditing & Cleaning:** Ingests and sanitizes over 500,000 raw transaction records, handling cancellations, refunds, missing customer identifiers, and pricing anomalies with a fully reproducible audit log.
2. **Relational Analytical Storage:** Automatically constructs a SQLite database with indexed tables and runs optimized SQL queries for business reporting (revenue trends, repeat purchase rates, country breakdowns).
3. **Behavioral RFM Analysis:** Evaluates customer Recency, Frequency, and Monetary metrics using quintile scoring.
4. **Unsupervised Customer Segmentation:** Uses K-Means clustering on standardized RFM features, selecting the optimal cluster count $K$ via Elbow Method (inertia) and Silhouette Score analysis, followed by rule-based algorithmic labeling (`Champions`, `Loyal Customers`, `At Risk`, `Potential`, `Lost Customers`).
5. **Supervised Propensity Modeling:** Trains a regularized Logistic Regression classifier with balanced class weighting to predict customer re-engagement. Crucially, a **strict temporal holdout window** is enforced to eliminate data leakage.
6. **Dynamic Business Insights:** Automatically extracts revenue concentration risks, churn warnings, and campaign priority groups from real calculated metrics.

---

## 💼 Alignment with Data Science Intern Job Descriptions

This project was intentionally engineered to demonstrate the exact competencies evaluated in Data Science / Data Analytics internship interviews:

| Target Skill | Implementation in this Project |
|---|---|
| **Python & Software Engineering** | Modular `src/` architecture, strict type hints (`typing`), unit tests (`unittest`), PEP 8 standards, and decoupled business logic. |
| **SQL & Database Querying** | Embedded SQLite database (`customer_analytics.db`), indexes on `CustomerID` and `InvoiceDate`, multi-table aggregations, and subqueries. |
| **Data Quality & Preprocessing** | Systematic data profiling, missingness diagnosis, cancellation filtering (`C` prefix handling), and validation logs. |
| **Exploratory Data Analysis (EDA)** | Interactive Plotly dashboards analyzing sales seasonality, customer purchase frequency, basket size distributions, and geographical concentration. |
| **Unsupervised ML (Clustering)** | Feature scaling (`StandardScaler`), cluster optimization (Elbow Method + Silhouette coefficients), and centroid interpretation. |
| **Supervised ML (Classification)** | Binary classification using Logistic Regression, temporal train/test splitting to prevent look-ahead bias, class imbalance handling, and evaluation via ROC-AUC, Precision-Recall, and Confusion Matrices. |
| **Business Communication** | Translates statistical findings into ROI-driven retention and re-engagement strategies with executive KPI cards. |

---

## 🏗️ Architecture & Project Structure

```text
customer-analytics-dashboard/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                     # Streamlit application entry point (6 interactive pages)
├── data/
│   ├── README.md              # Dataset documentation and manual download guide
│   └── customer_analytics.db  # SQLite database generated automatically
├── models/
│   └── README.md              # Model persistence documentation
├── notebooks/
│   └── customer_analysis.ipynb# Step-by-step Jupyter notebook mirroring the pipeline
├── screenshots/
│   └── README.md              # UI preview assets
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # UCI download, SQLite ingestion, and analytical SQL queries
│   ├── preprocessing.py      # Quality validation, cancellation handling, derived features
│   ├── eda.py                # Reusable Plotly chart generators for exploratory analysis
│   ├── rfm.py                # RFM computation and quintile scoring
│   ├── segmentation.py       # StandardScaler, Elbow/Silhouette analysis, K-Means clustering
│   ├── campaign_model.py     # Leakage-free temporal split, Logistic Regression, propensity scoring
│   └── insights.py           # Algorithmic business insight generator (no hardcoded metrics)
└── tests/
    └── test_pipeline.py       # Synthetic data test suite covering all modules
```

---

## 📊 Methodology & Technical Deep-Dive

### 1. Data Cleaning & Integrity Auditing
Raw transactions from the UCI Online Retail II dataset contain messy real-world artifacts:
- **Cancellations & Credit Memos:** Identified by invoice numbers starting with `'C'` and negative quantities. These are flagged and isolated to prevent skewing customer sales aggregates.
- **Missing Customer IDs:** Approximately 25% of records lack a `Customer ID` (guest checkouts). These transactions are tracked in the data quality audit report and excluded from customer-level RFM modeling.
- **Outliers & Anomalies:** Records with non-positive unit prices or extreme zero-quantity adjustments are removed.

### 2. Analytical SQL Engine
Rather than performing all aggregations in memory with Pandas, the cleaned data is written to an indexed SQLite table (`transactions`). Real SQL queries calculate:
- Monthly revenue and order volume trends
- Country-level revenue breakdowns
- Top 20 customers by total spend
- Repeat purchase frequency distributions
- Average Order Value (AOV)

### 3. RFM Calculation & K-Means Clustering
- **Recency ($R$):** Days elapsed between the customer's last purchase and the snapshot reference date ($\max(\text{InvoiceDate}) + 1\text{ day}$).
- **Frequency ($F$):** Total number of distinct completed invoices per customer.
- **Monetary ($M$):** Total net revenue generated by the customer.
- **Scaling:** Features are standardized using `StandardScaler` to prevent $M$ (spanning thousands of pounds) from dominating Euclidean distance calculations in K-Means.
- **Cluster Selection:** Tested $K \in [2, 8]$ computing both within-cluster sum of squares (Inertia) and Mean Silhouette Scores. The optimal $K \in [4, 6]$ is selected automatically.
- **Automated Labeling:** Centroid characteristics are evaluated algorithmically to assign business personas: `Champions`, `Loyal Customers`, `Potential Customers`, `At Risk`, and `Lost Customers`.

### 4. Leakage-Free Campaign Propensity Modeling
- **The Problem:** The UCI dataset does not include marketing campaign labels.
- **Analytical Proxy:** A supervised target was formulated: *Will an existing customer make a repeat purchase in the final 90 days of the observation period?*
- **Leakage Prevention:**
  - The dataset is split temporally into an **Observation Period** (prior to the final 90 days) and a **Holdout Period** (the final 90 days).
  - All predictive features ($R, F, M, \text{AOV}, \text{Basket Size}, \text{Product Variety}$) are computed **strictly from the Observation Period**.
  - The binary target $y \in \{0, 1\}$ is evaluated **strictly in the Holdout Period**.
  - `StandardScaler` is fitted solely on the training split and transformed on the test split.
- **Algorithm:** Logistic Regression with `class_weight='balanced'` to counter class imbalance. Outputs continuous propensity scores $P(\text{re-engagement}) \in [0, 1]$ used to prioritize outreach (`High Priority`: $\ge 0.65$, `Medium Priority`: $0.40 - 0.65$, `Low Priority`: $< 0.40$).

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-analytics-dashboard.git
cd customer-analytics-dashboard
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Automated Test Suite
```bash
python -m unittest discover tests
```
*Runs unit tests across data cleaning, RFM scoring, clustering, and modeling with synthetic data.*

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
*On first execution, the app automatically downloads the dataset (~6 MB) directly from the UCI Machine Learning Repository, caches it in `data/`, and populates the SQLite database.*

---

## 🌐 Deploying to Streamlit Community Cloud

1. Push your repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"** and select your repository and branch (`main`).
4. Set **Main file path** to: `app.py`.
5. Click **Deploy**. The app handles the dataset download and SQLite creation automatically on cold start.

---

## 📝 Resume Bullets (Ready to Copy)

Use these quantified, ATS-friendly bullet points on your resume for Data Science and Data Analytics roles:

> - **Engineered an end-to-end Customer Analytics & Campaign Intelligence platform** in Python, Streamlit, and SQLite, processing 500K+ transactional records from the UCI Online Retail II dataset.
> - **Built a behavioral customer segmentation engine** applying RFM analysis and K-Means clustering; determined optimal cluster count via Elbow Method and Silhouette analysis ($K=4\text{--}6$), identifying high-value revenue drivers and at-risk churn cohorts.
> - **Developed a campaign re-engagement propensity model** using regularized Logistic Regression and temporal holdout validation, eliminating data leakage and categorizing customers into actionable outreach tiers based on predicted conversion probability.
> - **Architected an embedded SQLite reporting pipeline** with parameterized analytical queries and indexes, powering real-time executive KPI cards and dynamic business insight generation in Streamlit.

---

## 💡 Technical Interview Q&A (Prepare for Interviews)

### Q1: Why did you scale features before running K-Means?
> **Answer:** K-Means is a distance-based clustering algorithm relying on Euclidean distance. In transactional data, Monetary value ranges from tens to thousands of pounds, whereas Frequency typically ranges from 1 to 50, and Recency from 0 to 365 days. Without normalization (such as `StandardScaler`), the Monetary dimension would completely dominate the distance calculation, making the clusters essentially a 1D grouping of spending rather than true multi-dimensional behavioral segments.

### Q2: How did you select the number of clusters $K$?
> **Answer:** Rather than guessing $K$, I combined the Elbow Method and Silhouette Analysis across candidate values $K \in [2, 8]$. The Elbow Method visualizes the diminishing returns in within-cluster inertia (sum of squared distances to centroids). Silhouette analysis measures cluster cohesion and separation (how close points are to their own cluster compared to neighboring clusters). I programmatically selected the $K$ that maximized the mean silhouette coefficient within a business-interpretable range ($K \in [4, 6]$).

### Q3: How did you prevent data leakage in your campaign propensity model?
> **Answer:** In time-series customer transaction data, calculating features (like Recency or Monetary) using the entire timeline while predicting an event within that same timeline introduces look-ahead bias and severe leakage. To prevent this, I applied a strict temporal holdout split: features were computed exclusively from historical transactions prior to the final 90 days, while the binary target (whether the customer made a purchase) was evaluated strictly within the final 90 days. Furthermore, all feature transformers (`StandardScaler`) were fitted exclusively on training fold data.

### Q4: Why use Logistic Regression over a Black-Box Model like XGBoost?
> **Answer:** For marketing campaign analytics, stakeholder interpretability and calibrated probability scores are critical. Logistic Regression coefficients provide direct insight into which customer behaviors (e.g., historical frequency vs basket size) drive repeat purchase odds. It serves as an optimal baseline model that prevents overfitting on tabular RFM features while generating well-calibrated propensity scores ($0.0 - 1.0$) needed for budget-constrained customer prioritization.

---

## 📜 Dataset Attribution & License

- **Dataset:** Online Retail II (UCI Machine Learning Repository)
- **Creators:** Dr. Daqing Chen, School of Engineering, London South Bank University
- **Licence:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **Citation:** Chen, D. (2019). Online Retail II. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
