"""
pull_trends.py

Nike Marketing ROI Analysis — Step 1: Pull Google Trends Data
Uses pytrends (unofficial Google Trends API, no key required)
"""
#-----------
# Imports
#-----------
import pandas as pd
import time
from pytrends.request import TrendReq

#-----------
# Setup
#-----------

pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))

KEYWORDS = ["Nike shoes", "Nike running", "Nike sale", "Nike Air Max"]
START    = "2022-01-01"
END      = "2023-09-30"
GEO      = "US"

#--------------------------------
# Pull weekly interest over time
#--------------------------------
# Google Trends allows max 5 keywords per request
print("Pulling Google Trends data …")
pytrends.build_payload(KEYWORDS, cat=0, timeframe=f"{START} {END}", geo=GEO)
trends_df = pytrends.interest_over_time()

if "isPartial" in trends_df.columns:
    trends_df = trends_df.drop(columns=["isPartial"])

trends_df.index.name = "week"
trends_df = trends_df.reset_index()

#-----------------------------------------------
# Create a composite "Nike brand interest" index
#-----------------------------------------------
trends_df["brand_index"] = trends_df[KEYWORDS].mean(axis=1).round(1)

#----------------------------------------------------
# Resample to quarterly for merge with financial data
#----------------------------------------------------
trends_df["week"] = pd.to_datetime(trends_df["week"])
trends_df = trends_df.set_index("week")

quarterly_trends = trends_df["brand_index"].resample("QE").mean().round(1)
quarterly_trends.index = quarterly_trends.index.to_period("Q")
quarterly_trends.name = "trends_index"

print(trends_df.head(10).to_string())
print("\nQuarterly average brand interest:")
print(quarterly_trends)

#-----------
# Save
#-----------

trends_df.to_csv("C:\\Users\\alica\\OneDrive\\Documents\\marketing-effectiveness-analysis\\data\\trends_weekly.csv")
quarterly_trends.to_csv("C:\\Users\\alica\\OneDrive\\Documents\\marketing-effectiveness-analysis\\data\\trends_quarterly.csv")
print("\n✓ Saved to data/trends_weekly.csv and data/trends_quarterly.csv")
