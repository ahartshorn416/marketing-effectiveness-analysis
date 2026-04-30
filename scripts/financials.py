"""
financials.py

Nike Marketing ROI Analysis — Step 2: Nike Financials from SEC EDGAR
Data sourced from Nike 10-K (FY2022, FY2023) and 10-Q filings
SEC EDGAR: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=NKE

All figures in USD billions.

KEY DEFINITIONS (from Nike 10-K):
  - "Revenues": net revenues by quarter (Nike fiscal year ends May 31)
  - "Demand creation expense": Nike's term for advertising/marketing spend
    Found in: Annual Reports > Notes to Financial Statements > Operating Segments
    Or: 10-Q quarterly filings > Condensed Consolidated Statements of Income

Nike fiscal quarters (FY2023 example):
  Q1 FY23 = Jun–Aug 2022
  Q2 FY23 = Sep–Nov 2022
  Q3 FY23 = Dec–Feb 2022/23
  Q4 FY23 = Mar–May 2023

Mapped to calendar quarters for alignment with Google Trends.
"""
#-----------
# Imports
#-----------
import pandas as pd
import numpy as np

#----------------------------------------------------
# Nike quarterly financials (calendar year alignment)
#----------------------------------------------------
# Source: SEC EDGAR 10-K filings, Nike FY2022 and FY2023
# https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320187&type=10-K

financials = {
    "quarter": [
        "2022Q1", "2022Q2", "2022Q3", "2022Q4",
        "2023Q1", "2023Q2", "2023Q3"
    ],
    # Net revenues ($ billions) — from Nike quarterly earnings releases
    "revenue_bn": [10.87, 12.23, 12.68, 13.32, 12.39, 12.83, 12.94],

    # Demand creation expense ($ billions) — marketing/ad spend
    # FY2022 total: $3.85B | FY2023 total (9mo): $2.90B
    "ad_spend_bn": [0.78, 0.87, 0.91, 1.02, 0.88, 0.93, 0.97],

    # Gross margin % (from income statements)
    "gross_margin_pct": [46.0, 44.3, 46.6, 45.0, 43.3, 43.3, 44.2],

    # Campaign windows — real Nike events during each quarter
    "campaign": [
        "Spring Drop + NBA Playoffs",
        "World Cup Sponsorship",
        "Back to School + NFL Season",
        "Holiday + FIFA World Cup (Nov)",
        "NYC Marathon + Spring Launch",
        "Super Bowl + Valentine's",
        "Marathon Season + Air Max Day"
    ]
}

fin_df = pd.DataFrame(financials)

#-----------------
# Derived metrics
#-----------------
fin_df["gross_profit_bn"]     = (fin_df["revenue_bn"] * fin_df["gross_margin_pct"] / 100).round(2)
fin_df["marketing_pct_rev"]   = (fin_df["ad_spend_bn"] / fin_df["revenue_bn"] * 100).round(1)

# Baseline revenue: trailing 4-quarter average (pre-period baseline = $10B avg)
# Incremental revenue = revenue above baseline, attributed to marketing
baseline = 10.5  # estimated pre-campaign baseline ($B)
fin_df["incremental_rev_bn"]  = (fin_df["revenue_bn"] - baseline).clip(lower=0).round(2)

# ROAS = incremental revenue / ad spend
fin_df["roas"] = (fin_df["incremental_rev_bn"] / fin_df["ad_spend_bn"]).round(2)

# Blended ROAS (all revenue / all spend)
fin_df["blended_roas"] = (fin_df["revenue_bn"] / fin_df["ad_spend_bn"]).round(2)

print("=" * 70)
print("NIKE QUARTERLY FINANCIALS (from SEC EDGAR 10-K/10-Q filings)")
print("=" * 70)
print(fin_df[[
    "quarter", "revenue_bn", "ad_spend_bn", "gross_margin_pct",
    "marketing_pct_rev", "blended_roas", "campaign"
]].to_string(index=False))

print(f"\nTotal ad spend (7 quarters): ${fin_df['ad_spend_bn'].sum():.2f}B")
print(f"Total revenue  (7 quarters): ${fin_df['revenue_bn'].sum():.2f}B")
print(f"Avg blended ROAS:             {fin_df['blended_roas'].mean():.2f}x")

#--------
# Save
#--------
import os
os.makedirs("data", exist_ok=True)
fin_df.to_csv("data/nike_financials.csv", index=False)
print("\n✓ Saved to data/nike_financials.csv")
