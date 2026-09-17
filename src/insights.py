"""
insights.py
-----------
Generates business insights from the analysis results.

All insight strings are constructed from ACTUAL calculated values.
No hard-coded fabricated statements.

Each insight is returned as a dict:
  {
    'title':          str   — short headline
    'observation':    str   — what the data shows
    'recommendation': str   — actionable business advice
    'severity':       str   — 'success' | 'warning' | 'info'
  }
"""

from typing import Dict, List, Optional

import pandas as pd


def generate_insights(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    scored: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    Produce a list of business insight dicts computed from the actual data.

    Parameters
    ----------
    df     : cleaned transactions DataFrame
    rfm    : RFM + Segment DataFrame
    scored : optional propensity-scored DataFrame
    """
    insights: List[Dict] = []

    total_rev = rfm["Monetary"].sum()
    n_customers = len(rfm)

    # ── 1. Revenue concentration by segment ──────────────────────────────────
    seg_rev = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)
    top_seg = seg_rev.index[0]
    top_pct = round(seg_rev.iloc[0] / total_rev * 100, 1)

    insights.append({
        "title": f"Revenue Concentration: '{top_seg}' Dominates",
        "observation": (
            f"The '{top_seg}' segment generates {top_pct}% of total revenue "
            f"(£{seg_rev.iloc[0]:,.0f} out of £{total_rev:,.0f})."
        ),
        "recommendation": (
            f"Invest in protecting the '{top_seg}' segment. "
            "Use loyalty programmes, exclusive offers, and personalised outreach "
            "to minimise churn risk in your highest-value group."
        ),
        "severity": "success" if top_pct < 60 else "warning",
    })

    # ── 2. At-Risk high-value customers ───────────────────────────────────────
    at_risk = rfm[rfm["Segment"] == "At Risk"]
    if not at_risk.empty:
        at_risk_rev = at_risk["Monetary"].sum()
        at_risk_pct = round(at_risk_rev / total_rev * 100, 1)
        at_risk_avg_m = at_risk["Monetary"].mean()
        insights.append({
            "title": "At-Risk Customers: Revenue at Stake",
            "observation": (
                f"{len(at_risk)} customers are flagged 'At Risk' — "
                f"they purchased historically (avg. £{at_risk_avg_m:,.0f} each) "
                f"but have become inactive. Together they represent "
                f"£{at_risk_rev:,.0f} ({at_risk_pct}%) of revenue."
            ),
            "recommendation": (
                "Launch a targeted win-back campaign for At Risk customers. "
                "Personalised emails with time-limited discounts or free shipping "
                "are typically effective. Prioritise those with highest Monetary value."
            ),
            "severity": "warning",
        })

    # ── 3. Customer concentration (top 10) ────────────────────────────────────
    top10_rev = rfm.nlargest(10, "Monetary")["Monetary"].sum()
    top10_pct = round(top10_rev / total_rev * 100, 1)
    insights.append({
        "title": f"Top 10 Customers = {top10_pct}% of Revenue",
        "observation": (
            f"The 10 highest-spending customers account for {top10_pct}% of total revenue. "
            f"{'This is a significant concentration risk.' if top10_pct > 30 else 'Revenue is reasonably diversified.'}"
        ),
        "recommendation": (
            "Ensure top accounts have dedicated relationship management. "
            "Simultaneously invest in growing mid-tier customers to reduce dependency risk."
        ),
        "severity": "warning" if top10_pct > 30 else "info",
    })

    # ── 4. Lost Customers ─────────────────────────────────────────────────────
    lost = rfm[rfm["Segment"] == "Lost Customers"]
    if not lost.empty:
        lost_pct = round(len(lost) / n_customers * 100, 1)
        insights.append({
            "title": "Lost Customers: Last-Resort Re-engagement",
            "observation": (
                f"{len(lost)} customers ({lost_pct}% of the base) are classified "
                "'Lost' — high recency, low engagement, low spend."
            ),
            "recommendation": (
                "Run one final re-engagement campaign. "
                "If no response, redirect that budget to acquisition. "
                "Consider a 'we miss you' email with a compelling incentive."
            ),
            "severity": "info",
        })

    # ── 5. Champions profile ──────────────────────────────────────────────────
    champs = rfm[rfm["Segment"] == "Champions"]
    if not champs.empty:
        champ_avg_m = champs["Monetary"].mean()
        champ_avg_f = champs["Frequency"].mean()
        champ_rev_pct = round(champs["Monetary"].sum() / total_rev * 100, 1)
        insights.append({
            "title": "Champions: Your Most Valuable Customers",
            "observation": (
                f"{len(champs)} Champions spend an average of £{champ_avg_m:,.0f} "
                f"and purchase {champ_avg_f:.1f} times on average, "
                f"contributing {champ_rev_pct}% of total revenue."
            ),
            "recommendation": (
                "Reward Champions with early access, VIP tiers, and referral bonuses. "
                "Their satisfaction directly translates to word-of-mouth growth."
            ),
            "severity": "success",
        })

    # ── 6. Seasonality ────────────────────────────────────────────────────────
    monthly_rev = df.groupby("YearMonth")["TotalPrice"].sum()
    if not monthly_rev.empty:
        peak_month = monthly_rev.idxmax()
        peak_val   = monthly_rev.max()
        trough_month = monthly_rev.idxmin()
        trough_val   = monthly_rev.min()
        ratio = round(peak_val / trough_val, 1) if trough_val > 0 else "N/A"
        insights.append({
            "title": "Revenue Seasonality Detected",
            "observation": (
                f"Peak revenue of £{peak_val:,.0f} occurred in {peak_month}. "
                f"The lowest month was {trough_month} at £{trough_val:,.0f} "
                f"(peak/trough ratio: {ratio}×)."
            ),
            "recommendation": (
                "Align marketing spend with high-demand months. "
                "Use low-demand periods for re-engagement, loyalty nurturing, "
                "and new-customer acquisition campaigns."
            ),
            "severity": "info",
        })

    # ── 7. Potential customers ────────────────────────────────────────────────
    potential = rfm[rfm["Segment"] == "Potential Customers"]
    if not potential.empty:
        pot_avg_f = potential["Frequency"].mean()
        insights.append({
            "title": "Growth Opportunity: Potential Customers",
            "observation": (
                f"{len(potential)} customers classified as 'Potential' have purchased "
                f"recently but only {pot_avg_f:.1f} times on average. "
                "They represent an up-sell and cross-sell opportunity."
            ),
            "recommendation": (
                "Send targeted product recommendations and bundle offers. "
                "A second purchase is the strongest predictor of long-term loyalty."
            ),
            "severity": "success",
        })

    # ── 8. Average order value insight ───────────────────────────────────────
    avg_ov = df.groupby("InvoiceNo")["TotalPrice"].sum().mean()
    insights.append({
        "title": f"Average Order Value: £{avg_ov:,.2f}",
        "observation": (
            f"The average basket size across all orders is £{avg_ov:,.2f}. "
            "Orders below this threshold may be opportunities for up-sell tactics."
        ),
        "recommendation": (
            "Implement 'Frequently bought together' recommendations and "
            f"a free-shipping threshold just above £{avg_ov:,.0f} to increase AOV."
        ),
        "severity": "info",
    })

    # ── 9. High-propensity re-engagement (model output) ──────────────────────
    if scored is not None and "Priority" in scored.columns:
        high_prio = scored[scored["Priority"] == "High Priority"]
        if not high_prio.empty:
            insights.append({
                "title": f"{len(high_prio)} High-Priority Re-engagement Targets",
                "observation": (
                    f"The propensity model identifies {len(high_prio)} customers "
                    "with a predicted re-engagement probability above 65%."
                ),
                "recommendation": (
                    "Focus immediate campaign budget on these High Priority customers. "
                    "Segment them further by their customer segment to tailor the message."
                ),
                "severity": "success",
            })

    return insights
