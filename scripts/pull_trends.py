"""
pull_trends.py

Nike Marketing ROI Analysis — Step 1: Pull Google Trends Data
Uses pytrends (unofficial Google Trends API, no key required)

Two keyword batches are pulled separately (Google Trends max 5 per request):
  Batch 1 — purchase intent:  buy, sale, discount, outlet, deals
  Batch 2 — product interest: Air Max, Dunk, Jordan, running shoes, sneakers

A weighted composite index is built from both batches:
  brand_index = 60% purchase intent + 40% product interest
"""
#-----------
# Imports
#-----------
import os
import pathlib
import pandas as pd
import time
from pytrends.request import TrendReq

#-----------
# Paths
#-----------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

#-----------
# Setup
#-----------
pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))

BATCH_1 = ["Nike buy", "Nike sale", "Nike discount", "Nike outlet", "Nike deals"]
BATCH_2 = ["Nike Air Max", "Nike Dunk", "Nike Jordan", "Nike running shoes", "Nike sneakers"]
START = "2022-01-01"
END = "2023-09-30"
GEO = "US"

#----------------------------------
# Batch 1: purchase intent keywords
#----------------------------------
print("Pulling Batch 1 — purchase intent keywords …")
pytrends.build_payload(BATCH_1, cat=0, timeframe=f"{START} {END}", geo=GEO)
df1 = pytrends.interest_over_time()
df1 = df1.drop(columns=["isPartial"], errors="ignore")
print(f"  Got {len(df1)} weeks")
print(df1.head(3).to_string())

print("\nWaiting 10s to avoid rate limiting …")
time.sleep(10)

# -------------------------------------
# Batch 2: product interest keywords
# -------------------------------------
print("\nPulling Batch 2 — product interest keywords …")
pytrends.build_payload(BATCH_2, cat=0, timeframe=f"{START} {END}", geo=GEO)
df2 = pytrends.interest_over_time()
df2 = df2.drop(columns=["isPartial"], errors="ignore")
print(f"  Got {len(df2)} weeks")
print(df2.head(3).to_string())

# --------------------------------------------
# Merge and build weighted composite index
# --------------------------------------------
trends_df = pd.concat([df1, df2], axis=1)
trends_df.index.name = "week"

# Purchase intent weighted higher — more directly predictive of revenue
purchase_intent = df1.mean(axis=1)
product_interest = df2.mean(axis=1)
trends_df["purchase_intent_index"] = purchase_intent.round(1)
trends_df["product_interest_index"] = product_interest.round(1)
trends_df["brand_index"] = (
        purchase_intent * 0.6 +
        product_interest * 0.4
).round(1)

# -------------------------
# Resample to quarterly
# -------------------------
quarterly_trends = trends_df["brand_index"].resample("QE").mean().round(1)
quarterly_trends.index = quarterly_trends.index.to_period("Q")
quarterly_trends.name = "trends_index"

# -----------------
# Print summary
# -----------------
print("\n" + "=" * 60)
print("COMPOSITE BRAND INDEX (60% purchase intent, 40% product)")
print("=" * 60)
print(trends_df[["purchase_intent_index", "product_interest_index", "brand_index"]].head(10).to_string())
print("\nQuarterly average brand index:")
print(quarterly_trends)

#--------
# Save
#--------
weekly_path = DATA_DIR / "trends_weekly.csv"
quarterly_path = DATA_DIR / "trends_quarterly.csv"

trends_df.to_csv(weekly_path)
quarterly_trends.to_csv(quarterly_path)

print(f"\n✓ Weekly data saved to   {weekly_path}")
print(f"✓ Quarterly data saved to {quarterly_path}")
