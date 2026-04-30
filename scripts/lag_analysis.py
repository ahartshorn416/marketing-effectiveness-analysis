"""
lag_analysis.py

Nike Marketing ROI Analysis — Step 3: Lag Correlation Analysis
Finds the empirical lag between Google Trends spikes and revenue lifts.

Key insight: If Trends search volume for "Nike" spikes in week 1,
how many weeks later does revenue actually increase?

Method: Cross-correlation between weekly Trends index and 
rolling 4-week revenue proxy (interpolated from quarterly data).
"""
#------------
# Imports
#------------
import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats

# --------
# Paths
# --------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print(f"Saving outputs to: {RESULTS_DIR}\n")

#--------------------------------------------------
# Load real Google Trends data from pull_trends.py
#--------------------------------------------------
trends_csv = PROJECT_ROOT / "data" / "trends_weekly.csv"

if trends_csv.exists():
    print(f"Loading real Trends data from {trends_csv}")
    trends_df = pd.read_csv(trends_csv, index_col=0, parse_dates=True)
    trends_weekly = trends_df["brand_index"].dropna()
    date_range = trends_weekly.index
    n = len(trends_weekly)
    print(f"Loaded {n} weeks of real Trends data ({date_range[0].date()} → {date_range[-1].date()})\n")
else:
    print("WARNING: data/trends_weekly.csv not found — using simulated data.")
    print("Run pull_trends.py first to use real Google Trends data.\n")
    np.random.seed(42)
    date_range = pd.date_range("2022-01-03", "2023-09-25", freq="W-MON")
    n = len(date_range)
    trends_base = 45 + 10 * np.sin(np.linspace(0, 4 * np.pi, n))
    spikes = np.zeros(n)
    for center, height in [(22, 28), (48, 35), (75, 20), (16, 18), (64, 22), (88, 15)]:
        if center < n:
            spikes += height * np.exp(-0.5 * ((np.arange(n) - center) / 3) ** 2)
    trends_weekly = pd.Series(
        np.clip(trends_base + spikes + np.random.normal(0, 3, n), 20, 100),
        index=date_range, name="brand_index"
    )

# ------------------------------------------
# Interpolate quarterly revenue to weekly
# ------------------------------------------
revenue_quarterly = pd.Series(
    [10.87, 12.23, 12.68, 13.32, 12.39, 12.83, 12.94],
    index=pd.period_range("2022Q1", periods=7, freq="Q")
)

# Convert quarter start dates to numeric timestamps for interpolation
quarter_dates = revenue_quarterly.index.to_timestamp().astype(np.int64)
weekly_dates  = date_range.astype(np.int64)

# Use numpy linear interpolation — no reindex, no alignment issues
rev_values = np.interp(weekly_dates, quarter_dates, revenue_quarterly.values)
rev_weekly = pd.Series(rev_values, index=date_range)

# --------------------------
# Debug: confirm no NaNs
# --------------------------
print(f"trends_weekly length: {len(trends_weekly)}, NaNs: {trends_weekly.isna().sum()}")
print(f"rev_weekly    length: {len(rev_weekly)},    NaNs: {rev_weekly.isna().sum()}")

if rev_weekly.isna().sum() > 0 or trends_weekly.isna().sum() > 0:
    print("WARNING: NaNs detected — filling remaining gaps.")
    rev_weekly = rev_weekly.ffill().bfill()
    trends_weekly = trends_weekly.ffill().bfill()

#--------------------------------------
# Cross-correlation at lags 0-10 weeks
#--------------------------------------
max_lag = 12
correlations = []

for lag in range(0, max_lag + 1):
    a = trends_weekly.values[:-lag] if lag > 0 else trends_weekly.values
    b = rev_weekly.values[lag:] if lag > 0 else rev_weekly.values

    # Drop any remaining NaN pairs
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if len(a) < 10:
        correlations.append({"lag_weeks": lag, "correlation": np.nan, "p_value": np.nan})
        continue

    r, p = stats.pearsonr(a, b)
    correlations.append({"lag_weeks": lag, "correlation": round(r, 3), "p_value": round(p, 4)})

corr_df = pd.DataFrame(correlations)
corr_df_valid = corr_df.dropna(subset=["correlation"])

print("\n" + "=" * 55)
print("CROSS-CORRELATION: Google Trends → Revenue")
print("=" * 55)
print(corr_df.to_string(index=False))

if corr_df_valid.empty:
    print("\nERROR: All correlations are NaN. Check debug output above.")
    exit(1)

best_lag = corr_df_valid.loc[corr_df_valid["correlation"].idxmax()]
optimal_lag = int(best_lag["lag_weeks"])

print(f"\n✓ Best lag: {optimal_lag} weeks "
      f"(r = {best_lag['correlation']}, p = {best_lag['p_value']})")
print(f"  Interpretation: Trends spikes predict revenue lifts ~{optimal_lag} weeks later")

#------------
# Plot
#------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("Nike: Google Trends → Revenue Lag Analysis", fontsize=14, fontweight="bold")

# Panel 1: Trends vs Revenue with optimal lag applied
ax1 = axes[0]
ax1.plot(trends_weekly.index, trends_weekly.values, color="#185FA5", lw=1.5,
         label="Google Trends brand index", alpha=0.85)
ax1_r = ax1.twinx()
shifted_rev = rev_weekly.values[optimal_lag:] if optimal_lag > 0 else rev_weekly.values
shifted_idx = date_range[:len(shifted_rev)]
ax1_r.plot(shifted_idx, shifted_rev, color="#3B6D11", lw=2, linestyle="--",
           label=f"Revenue (shifted back {optimal_lag} wks)")
ax1.set_ylabel("Trends index", color="#185FA5", fontsize=11)
ax1_r.set_ylabel("Revenue ($B)", color="#3B6D11", fontsize=11)
ax1.set_title(f"Trends index vs Revenue (revenue shifted -{optimal_lag} weeks for alignment)",
              fontsize=11)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# Panel 2: Correlation by lag bar chart
ax2 = axes[1]
colors = ["#185FA5" if i == optimal_lag else "#B5D4F4" for i in corr_df["lag_weeks"]]
bars = ax2.bar(corr_df["lag_weeks"], corr_df["correlation"].fillna(0),
               color=colors, edgecolor="white", lw=0.5)
ax2.set_xlabel("Lag (weeks)", fontsize=11)
ax2.set_ylabel("Pearson correlation (r)", fontsize=11)
ax2.set_title("Correlation strength by lag — peak shows optimal predictive window", fontsize=11)
ax2.axhline(0, color="#888", lw=0.8)
ax2.set_xticks(corr_df["lag_weeks"])
for bar, row in zip(bars, corr_df.itertuples()):
    val = row.correlation if not np.isnan(row.correlation) else 0
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
             f"{val:.2f}", ha="center", va="bottom", fontsize=8,
             color="#0C447C" if row.lag_weeks == optimal_lag else "#555")
ax2.grid(axis="y", alpha=0.3)

#------------
# Save
#------------
chart_path = RESULTS_DIR / "lag_correlation.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Chart saved to {chart_path}")

csv_path = RESULTS_DIR / "lag_correlations.csv"
corr_df.to_csv(csv_path, index=False)
print(f"✓ Data saved to {csv_path}")