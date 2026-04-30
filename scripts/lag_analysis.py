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

# ---------------------------------------
# Aggregate weekly Trends to quarterly
# ---------------------------------------
trends_quarterly = trends_weekly.resample("QE").mean().round(1)
trends_quarterly.index = trends_quarterly.index.to_period("Q")

# -----------------------------------------------
# Nike quarterly revenue (SEC EDGAR 10-K/10-Q)
# -----------------------------------------------
revenue_quarterly = pd.Series(
    [10.87, 12.23, 12.68, 13.32, 12.39, 12.83, 12.94],
    index=pd.period_range("2022Q1", periods=7, freq="Q")
)

# ------------------------------------
# Align both series on shared quarters
# ------------------------------------
shared_idx = trends_quarterly.index.intersection(revenue_quarterly.index)
trends_q = trends_quarterly.loc[shared_idx]
revenue_q = revenue_quarterly.loc[shared_idx]

print(f"Quarterly aligned data ({len(shared_idx)} quarters):")
summary = pd.DataFrame({
    "trends_index": trends_q,
    "revenue_bn": revenue_q
})
print(summary.to_string())

#-----------------------------------------
# Cross-correlation at lags 0–4 quarters
#-----------------------------------------
max_lag = 4
correlations = []

for lag in range(0, max_lag + 1):
    a = trends_q.values[:-lag] if lag > 0 else trends_q.values
    b = revenue_q.values[lag:] if lag > 0 else revenue_q.values

    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if len(a) < 3:
        correlations.append({"lag_quarters": lag, "correlation": np.nan, "p_value": np.nan})
        continue

    r, p = stats.pearsonr(a, b)
    correlations.append({"lag_quarters": lag, "correlation": round(r, 3), "p_value": round(p, 4)})

corr_df = pd.DataFrame(correlations)
corr_df_valid = corr_df.dropna(subset=["correlation"])

print("\n" + "=" * 55)
print("CROSS-CORRELATION: Google Trends → Revenue (quarterly)")
print("=" * 55)
print(corr_df.to_string(index=False))

if corr_df_valid.empty:
    print("\nERROR: All correlations are NaN. Check debug output above.")
    exit(1)

best_lag = corr_df_valid.loc[corr_df_valid["correlation"].idxmax()]
optimal_lag = int(best_lag["lag_quarters"])

print(f"\n✓ Best lag: {optimal_lag} quarter(s) "
      f"(r = {best_lag['correlation']}, p = {best_lag['p_value']})")
print(f"  Interpretation: Trends spikes predict revenue lifts ~{optimal_lag} quarter(s) later")

# --------
# Plot
# --------
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("Nike: Google Trends → Revenue Lag Analysis (Quarterly)",
             fontsize=14, fontweight="bold")

# Panel 1: Trends vs Revenue overlay
ax1 = axes[0]
x = np.arange(len(shared_idx))
ax1.plot(x, trends_q.values, "o-", color="#185FA5", lw=2, ms=7,
         label="Trends brand index (quarterly avg)")
ax1_r = ax1.twinx()
ax1_r.plot(x, revenue_q.values, "s--", color="#3B6D11", lw=2, ms=7,
           label="Revenue ($B)")
ax1.set_xticks(x)
ax1.set_xticklabels([str(q) for q in shared_idx], fontsize=9)
ax1.set_ylabel("Trends index", color="#185FA5", fontsize=11)
ax1_r.set_ylabel("Revenue ($B)", color="#3B6D11", fontsize=11)
ax1.set_title("Quarterly Trends Index vs Nike Revenue — same-period overlay",
              fontsize=11)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# Annotate each quarter
for i, (t, r) in enumerate(zip(trends_q.values, revenue_q.values)):
    ax1.annotate(f"{t:.0f}", (i, t), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8, color="#185FA5")
    ax1_r.annotate(f"${r:.1f}B", (i, r), textcoords="offset points",
                   xytext=(0, -14), ha="center", fontsize=8, color="#3B6D11")

# Panel 2: Correlation by lag
ax2 = axes[1]
colors = ["#185FA5" if i == optimal_lag else "#B5D4F4"
          for i in corr_df["lag_quarters"]]
bars = ax2.bar(corr_df["lag_quarters"], corr_df["correlation"].fillna(0),
               color=colors, edgecolor="white", lw=0.5, width=0.5)
ax2.set_xlabel("Lag (quarters)", fontsize=11)
ax2.set_ylabel("Pearson correlation (r)", fontsize=11)
ax2.set_title("Correlation strength by lag — peak = optimal predictive window",
              fontsize=11)
ax2.axhline(0, color="#888", lw=0.8)
ax2.set_xticks(corr_df["lag_quarters"])
ax2.set_xticklabels([f"Q+{l}" for l in corr_df["lag_quarters"]], fontsize=10)
for bar, row in zip(bars, corr_df.itertuples()):
    val = row.correlation if not np.isnan(row.correlation) else 0
    offset = 0.01 if val >= 0 else -0.03
    ax2.text(bar.get_x() + bar.get_width() / 2, val + offset,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9,
             color="#0C447C" if row.lag_quarters == optimal_lag else "#555")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()

chart_path = RESULTS_DIR / "lag_correlation.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Chart saved to {chart_path}")

csv_path = RESULTS_DIR / "lag_correlations.csv"
corr_df.to_csv(csv_path, index=False)
print(f"✓ Data saved to {csv_path}")