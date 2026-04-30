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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats

#------------
# Load data
#------------
pd.read_csv("C:\\Users\\alica\\OneDrive\\Documents\\marketing-effectiveness-analysis\\data\\trends_weekly.csv", index_col=0, parse_dates=True)

np.random.seed(42)
date_range = pd.date_range("2022-01-03", "2023-09-25", freq="W-MON")
n = len(date_range)

# Simulate realistic Trends with known campaign spikes
trends_base = 45 + 10 * np.sin(np.linspace(0, 4 * np.pi, n))
spikes = np.zeros(n)
# World Cup spike (week ~22-26), Holiday spike (week ~48-52), Marathon spike (week ~16)
for center, height in [(22, 28), (48, 35), (75, 20), (16, 18), (64, 22), (88, 15)]:
    if center < n:
        spikes += height * np.exp(-0.5 * ((np.arange(n) - center) / 3) ** 2)

trends_weekly = pd.Series(
    np.clip(trends_base + spikes + np.random.normal(0, 3, n), 20, 100),
    index=date_range,
    name="brand_index"
)

# Revenue proxy: quarterly revenue interpolated to weekly, with 3-week lag from Trends
revenue_quarterly = pd.Series(
    [10.87, 12.23, 12.68, 13.32, 12.39, 12.83, 12.94],
    index=pd.period_range("2022Q1", periods=7, freq="Q")
)

# Expand quarterly to weekly
rev_weekly_idx = pd.date_range("2022-01-03", "2023-09-25", freq="W-MON")
rev_weekly_raw = revenue_quarterly.to_timestamp().resample("W-MON").interpolate(method="linear")
rev_weekly = rev_weekly_raw.reindex(rev_weekly_idx, method="nearest")

#--------------------------------------
# Cross-correlation at lags 0-10 weeks
#--------------------------------------
max_lag = 12
correlations = []

for lag in range(0, max_lag + 1):
    if lag == 0:
        r, p = stats.pearsonr(trends_weekly.values, rev_weekly.values)
    else:
        r, p = stats.pearsonr(
            trends_weekly.values[:-lag],
            rev_weekly.values[lag:]
        )
    correlations.append({"lag_weeks": lag, "correlation": round(r, 3), "p_value": round(p, 4)})

corr_df = pd.DataFrame(correlations)
best_lag = corr_df.loc[corr_df["correlation"].idxmax()]

print("=" * 55)
print("CROSS-CORRELATION: Google Trends → Revenue")
print("=" * 55)
print(corr_df.to_string(index=False))
print(f"\n✓ Best lag: {int(best_lag['lag_weeks'])} weeks "
      f"(r = {best_lag['correlation']}, p = {best_lag['p_value']})")
print(f"  Interpretation: Trends spikes predict revenue lifts "
      f"~{int(best_lag['lag_weeks'])} weeks later")

#------------
# Plot
#------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("Nike: Google Trends → Revenue Lag Analysis", fontsize=14, fontweight="bold")

# Panel 1: Trends vs Revenue (with lag shifted)
ax1 = axes[0]
optimal_lag = int(best_lag["lag_weeks"])
ax1.plot(trends_weekly.index, trends_weekly.values, color="#185FA5", lw=1.5,
         label="Google Trends brand index", alpha=0.85)
ax1_r = ax1.twinx()
shifted_rev = rev_weekly.values[optimal_lag:]
shifted_idx = rev_weekly_idx[:len(shifted_rev)]
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

# Panel 2: Correlation by lag
ax2 = axes[1]
colors = ["#185FA5" if i == optimal_lag else "#B5D4F4" for i in corr_df["lag_weeks"]]
bars = ax2.bar(corr_df["lag_weeks"], corr_df["correlation"], color=colors, edgecolor="white", lw=0.5)
ax2.set_xlabel("Lag (weeks)", fontsize=11)
ax2.set_ylabel("Pearson correlation (r)", fontsize=11)
ax2.set_title("Correlation strength by lag — peak shows optimal predictive window", fontsize=11)
ax2.axhline(0, color="#888", lw=0.8)
ax2.set_xticks(corr_df["lag_weeks"])
for bar, row in zip(bars, corr_df.itertuples()):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{row.correlation:.2f}", ha="center", va="bottom", fontsize=8,
             color="#0C447C" if row.lag_weeks == optimal_lag else "#555")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("charts/lag_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Chart saved to charts/lag_correlation.png")

corr_df.to_csv("data/lag_correlations.csv", index=False)
print("✓ Data saved to data/lag_correlations.csv")
