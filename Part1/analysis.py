"""
Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis
Forest Fires Dataset (Montesinho Natural Park, Portugal)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot aesthetics
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

SEPARATOR = "\n" + "=" * 80 + "\n"

# =============================================================================
# TASK 1 — Load dataset, print first 5 rows, dtypes, shape
# =============================================================================
print(SEPARATOR + "TASK 1: Load Dataset" + SEPARATOR)

df = pd.read_csv("forestfires.csv")

print("First 5 rows:")
print(df.head().to_string())

print("\nColumn data types:")
print(df.dtypes)

print(f"\nDataFrame shape: {df.shape}  ({df.shape[0]} rows × {df.shape[1]} columns)")

# =============================================================================
# TASK 2 — Null value analysis
# =============================================================================
print(SEPARATOR + "TASK 2: Null Value Analysis" + SEPARATOR)

null_count = df.isnull().sum()
null_pct = (df.isnull().sum() / df.shape[0]) * 100

null_table = pd.DataFrame({
    "Null Count": null_count,
    "Null Percentage (%)": null_pct.round(2)
})
print("Null values per column:")
print(null_table.to_string())

# Identify columns exceeding 20% null rate
cols_above_20 = null_pct[null_pct > 20].index.tolist()
if cols_above_20:
    print(f"\nColumns exceeding 20% null rate: {cols_above_20}")
else:
    print("\nNo columns exceed a 20% null rate.")

# Fill numeric columns below 20% nulls with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if 0 < null_pct[col] <= 20:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled {col} nulls with median = {median_val}")

print("\nAfter median imputation, remaining nulls:")
print(df.isnull().sum().to_string())

print("\n[README NOTE] Although the dataset contains no missing values, the required")
print("median-imputation logic was implemented to demonstrate the preprocessing")
print("strategy that would be applied if missing values were present. Median was")
print("chosen because it is robust to outliers and skewed distributions.")

# =============================================================================
# TASK 3 — Duplicate detection and removal
# =============================================================================
print(SEPARATOR + "TASK 3: Duplicate Detection and Removal" + SEPARATOR)

dup_count_before = df.duplicated().sum()
print(f"Duplicate rows found: {dup_count_before}")

null_pct_before_drop = (df.isnull().sum() / df.shape[0]) * 100

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

rows_removed = dup_count_before
print(f"Rows removed: {rows_removed}")
print(f"Shape after removal: {df.shape}")

null_pct_after_drop = (df.isnull().sum() / df.shape[0]) * 100
pct_change = null_pct_after_drop - null_pct_before_drop

if (pct_change.abs() > 0.001).any():
    print("\nNull percentage changes after duplicate removal:")
    changed = pct_change[pct_change.abs() > 0.001]
    for col in changed.index:
        print(f"  {col}: {null_pct_before_drop[col]:.2f}% → {null_pct_after_drop[col]:.2f}%")
else:
    print("\nDuplicate removal did not change any column's null percentage.")

# =============================================================================
# TASK 4 — Data type correction
# =============================================================================
print(SEPARATOR + "TASK 4: Data Type Correction" + SEPARATOR)

print("Current dtypes:")
print(df.dtypes)
print()

mem_before = df.memory_usage(deep=True).sum()
print(f"Memory usage BEFORE dtype conversion: {mem_before:,} bytes ({mem_before / 1024:.1f} KB)")

# The 'month' and 'day' columns are repetitive strings → convert to category
# X and Y are spatial coordinates stored as int64 but are really categorical
# (discrete grid positions 1–9 and 2–9)
print("\nConversions applied:")

# Convert month and day to category dtype (repetitive string columns)
df["month"] = df["month"].astype("category")
print("  • month: object → category")

df["day"] = df["day"].astype("category")
print("  • day: object → category")

# X and Y are discrete spatial grid coordinates (1-9). Although they represent
# locations, they were retained as numeric because distance and correlation 
# analyses require numeric representation.
#
# The dataset contains no numeric columns incorrectly stored as object, therefore 
# no numeric dtype correction was necessary. The required dtype optimization was 
# demonstrated by converting the repetitive string columns above.
# We'll still do a safe conversion pass just to confirm:
for col in df.columns:
    if df[col].dtype == "object":
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null_original = df[col].notna().sum()
        non_null_converted = converted.notna().sum()
        if non_null_converted == non_null_original:
            df[col] = converted
            print(f"  • {col}: object → numeric (via pd.to_numeric)")

mem_after = df.memory_usage(deep=True).sum()
print(f"\nMemory usage AFTER dtype conversion: {mem_after:,} bytes ({mem_after / 1024:.1f} KB)")
print(f"Memory saved: {mem_before - mem_after:,} bytes ({(mem_before - mem_after) / mem_before * 100:.1f}%)")

print("\nUpdated dtypes:")
print(df.dtypes)

# =============================================================================
# TASK 5 — Descriptive statistics and skewness
# =============================================================================
print(SEPARATOR + "TASK 5: Descriptive Statistics and Skewness" + SEPARATOR)

numeric_cols = df.select_dtypes(include=[np.number]).columns

print("Descriptive statistics (all numeric columns):")
print(df[numeric_cols].describe().round(3).to_string())

print("\nSkewness per numeric column:")
skewness = df[numeric_cols].skew().sort_values(key=abs, ascending=False)
print(skewness.round(4).to_string())

most_skewed_col = skewness.abs().idxmax()
most_skewed_val = skewness[most_skewed_col]
print(f"\nColumn with highest absolute skewness: '{most_skewed_col}' (skew = {most_skewed_val:.4f})")

if most_skewed_val > 0:
    print(f"  → Positive skew: distribution has a long right tail (many small values,")
    print(f"    few very large values). The mean is pulled above the median by extreme")
    print(f"    high values, making it a poor choice for imputation.")
else:
    print(f"  → Negative skew: distribution has a long left tail (many large values,")
    print(f"    few very small values). The mean is pulled below the median by extreme")
    print(f"    low values, making it a poor choice for imputation.")

# Also identify top 2 for later use in Task 9a
top2_skewed = skewness.abs().nlargest(2).index.tolist()
print(f"\nTop 2 most skewed columns: {top2_skewed}")

# =============================================================================
# TASK 6 — Outlier detection with IQR
# =============================================================================
print(SEPARATOR + "TASK 6: Outlier Detection with IQR" + SEPARATOR)

# Analyze the two most interesting columns: 'area' (target, highly skewed) and 'ISI'
iqr_cols = ["area", "ISI"]

iqr_results = {}
for col in iqr_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    n_outliers = len(outliers)
    iqr_results[col] = {
        "Q1": Q1, "Q3": Q3, "IQR": IQR,
        "Lower": lower, "Upper": upper,
        "Outlier Count": n_outliers,
        "Outlier %": round(n_outliers / len(df) * 100, 2)
    }

    print(f"\n--- {col} ---")
    print(f"  Q1 = {Q1:.2f},  Q3 = {Q3:.2f},  IQR = {IQR:.2f}")
    print(f"  Lower bound = {lower:.2f},  Upper bound = {upper:.2f}")
    print(f"  Outliers: {n_outliers} rows ({iqr_results[col]['Outlier %']}%)")
    if n_outliers > 0:
        print(f"  Outlier range: [{outliers[col].min():.2f}, {outliers[col].max():.2f}]")

print("\n[README NOTE] Outliers will NOT be dropped. They will be retained for modeling")
print("because extreme fire events are genuine observations, not measurement errors.")
print("In Part 2, a log transform on 'area' (ln(area+1)) will compress extreme values")
print("and reduce their influence without discarding real data.")

# =============================================================================
# TASK 7 — Visualizations (5 types)
# =============================================================================
print(SEPARATOR + "TASK 7: Visualizations" + SEPARATOR)

# --- 7a. Line Plot ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df["temp"], color="#2196F3", linewidth=0.8, alpha=0.8)
ax.set_title("Temperature Across Observations (Row Index)", fontsize=14, fontweight="bold")
ax.set_xlabel("Row Index (observation order)")
ax.set_ylabel("Temperature (°C)")
ax.grid(True, alpha=0.3)
plt.savefig("plot_1_lineplot.png")
plt.close()
print("  ✓ Line plot saved: plot_1_lineplot.png")

# --- 7b. Bar Chart ---
month_order = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
mean_area_by_month = df.groupby("month", observed=True)["area"].mean().reindex(month_order).dropna()

fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("YlOrRd", n_colors=len(mean_area_by_month))
ax.bar(mean_area_by_month.index, mean_area_by_month.values, color=colors, edgecolor="black", linewidth=0.5)
ax.set_title("Mean Burned Area by Month", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Mean Burned Area (ha)")
plt.savefig("plot_2_barchart.png")
plt.close()
print("  ✓ Bar chart saved: plot_2_barchart.png")

# --- 7c. Histogram of most skewed column ---
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df[most_skewed_col], bins=20, kde=True, color="#E91E63", ax=ax)
ax.set_title(f"Distribution of '{most_skewed_col}' (Skew = {most_skewed_val:.2f})",
             fontsize=14, fontweight="bold")
ax.set_xlabel(most_skewed_col)
ax.set_ylabel("Frequency")
plt.savefig("plot_3_histogram.png")
plt.close()
print(f"  ✓ Histogram saved: plot_3_histogram.png (column: {most_skewed_col})")

# --- 7d. Scatter Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x="temp", y="RH", data=df, alpha=0.6, color="#673AB7", ax=ax)
ax.set_title("Temperature vs. Relative Humidity", fontsize=14, fontweight="bold")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Relative Humidity (%)")
# Add correlation annotation
r = df["temp"].corr(df["RH"])
ax.annotate(f"Pearson r = {r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
            fontsize=12, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
plt.savefig("plot_4_scatterplot.png")
plt.close()
print("  ✓ Scatter plot saved: plot_4_scatterplot.png")

# --- 7e. Box Plot ---
fig, ax = plt.subplots(figsize=(12, 6))
order = ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"]
present_months = [m for m in order if m in df["month"].cat.categories]
sns.boxplot(x="month", y="temp", data=df, order=present_months,
            palette="coolwarm", ax=ax)
ax.set_title("Temperature Distribution by Month", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Temperature (°C)")
plt.savefig("plot_5_boxplot.png")
plt.close()
print("  ✓ Box plot saved: plot_5_boxplot.png")

print("\nAll 5 visualizations generated.")

# =============================================================================
# TASK 8 — Correlation Heat Map (Pearson)
# =============================================================================
print(SEPARATOR + "TASK 8: Pearson Correlation Heat Map" + SEPARATOR)

pearson_corr = df[numeric_cols].corr(method="pearson")

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            mask=mask, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Pearson Correlation Matrix — Numeric Variables", fontsize=14, fontweight="bold")
plt.savefig("plot_6_correlation_heatmap.png")
plt.close()
print("  ✓ Correlation heatmap saved: plot_6_correlation_heatmap.png")

# Find pair with highest absolute correlation (excluding self-correlations)
corr_unstacked = pearson_corr.where(
    np.triu(np.ones(pearson_corr.shape, dtype=bool), k=1)
).stack()
max_corr_pair = corr_unstacked.abs().idxmax()
max_corr_val = corr_unstacked[max_corr_pair]
print(f"\nHighest absolute Pearson correlation: {max_corr_pair[0]} ↔ {max_corr_pair[1]}"
      f" (r = {max_corr_val:.4f})")

print("\n[README NOTE] This correlation does not necessarily imply causation.")
print("A third variable (e.g., seasonal trends, geographic factors) could explain")
print("the observed relationship.")

# =============================================================================
# TASK 9a — Imputation Strategy Comparison
# =============================================================================
print(SEPARATOR + "TASK 9a: Imputation Strategy Comparison" + SEPARATOR)

print(f"Two columns with highest absolute skewness: {top2_skewed}\n")

print(f"{'Column':<12} {'Mean':>12} {'Median':>12} {'Skewness':>12} {'Chosen Statistic':<20}")
print("-" * 68)
for col in top2_skewed:
    col_mean = df[col].mean()
    col_median = df[col].median()
    col_skew = df[col].skew()
    # For skewed distributions, median is more representative
    chosen = "Median"
    print(f"{col:<12} {col_mean:>12.4f} {col_median:>12.4f} {col_skew:>12.4f} {chosen:<20}")

print("\nJustification:")
for col in top2_skewed:
    col_skew = df[col].skew()
    if col_skew > 0:
        print(f"  • {col} (skew={col_skew:.4f}): Positively skewed → the mean ({df[col].mean():.4f})")
        print(f"    is pulled upward by extreme high values. The median ({df[col].median():.4f})")
        print(f"    better represents the typical observation. → Use MEDIAN for imputation.")
    else:
        print(f"  • {col} (skew={col_skew:.4f}): Negatively skewed → the mean ({df[col].mean():.4f})")
        print(f"    is pulled downward by extreme low values. The median ({df[col].median():.4f})")
        print(f"    better represents the typical observation. → Use MEDIAN for imputation.")

# Apply median imputation to any remaining nulls in these two columns
for col in top2_skewed:
    df[col].fillna(df[col].median(), inplace=True)

print(f"\nAfter imputation, null counts for {top2_skewed}:")
for col in top2_skewed:
    print(f"  {col}: {df[col].isnull().sum()} nulls")

# =============================================================================
# TASK 9b — Spearman Rank Correlation
# =============================================================================
print(SEPARATOR + "TASK 9b: Spearman Rank Correlation" + SEPARATOR)

spearman_corr = df[numeric_cols].corr(method="spearman")

print("Spearman Correlation Matrix:")
print(spearman_corr.round(4).to_string())

# Compute |Spearman - Pearson| difference
diff = (spearman_corr - pearson_corr).abs()

# Extract upper triangle pairs
diff_unstacked = diff.where(
    np.triu(np.ones(diff.shape, dtype=bool), k=1)
).stack()

# Top 3 pairs with largest difference
top3_diff = diff_unstacked.nlargest(3)

print("\n\nTop 3 Column Pairs with Largest |Spearman − Pearson| Difference:")
print(f"{'Pair':<25} {'Pearson':>10} {'Spearman':>10} {'|Difference|':>14}")
print("-" * 60)
for (col_a, col_b), diff_val in top3_diff.items():
    p = pearson_corr.loc[col_a, col_b]
    s = spearman_corr.loc[col_a, col_b]
    print(f"{col_a} ↔ {col_b:<12} {p:>10.4f} {s:>10.4f} {diff_val:>14.4f}")

print("\nFull |Spearman − Pearson| Difference Table:")
print(diff.round(4).to_string())

print("\nInterpretation:")
for (col_a, col_b), diff_val in top3_diff.items():
    p_val = abs(pearson_corr.loc[col_a, col_b])
    s_val = abs(spearman_corr.loc[col_a, col_b])
    if s_val > p_val:
        rel_type = "monotonic but NON-LINEAR"
        explanation = (f"|Spearman| ({s_val:.4f}) > |Pearson| ({p_val:.4f}): the variables "
                       f"move together consistently but not proportionally.")
    else:
        rel_type = "approximately LINEAR"
        explanation = (f"|Pearson| ({p_val:.4f}) ≥ |Spearman| ({s_val:.4f}): the relationship "
                       f"is well captured by a linear model.")
    print(f"  • {col_a} ↔ {col_b}: {rel_type}")
    print(f"    {explanation}")

print("\n[README NOTE] For Part 2 feature selection:")
print("  Spearman correlation will be used as the primary guide because many variables")
print("  in this dataset have non-linear relationships (due to skewness and threshold")
print("  effects in fire behavior). Spearman captures monotonic relationships that")
print("  Pearson would underestimate.")

# =============================================================================
# TASK 9c — Grouped Aggregation
# =============================================================================
print(SEPARATOR + "TASK 9c: Grouped Aggregation" + SEPARATOR)

cat_col = "month"
num_col = "area"

grouped = df.groupby(cat_col, observed=True)[num_col].agg(["mean", "std", "count"])
# Reorder by calendar month
grouped = grouped.reindex([m for m in month_order if m in grouped.index])
print(f"Grouped aggregation: {num_col} by {cat_col}")
print(grouped.round(4).to_string())

highest_mean_group = grouped["mean"].idxmax()
highest_std_group = grouped["std"].idxmax()
highest_mean_val = grouped["mean"].max()
lowest_mean_val = grouped["mean"][grouped["mean"] > 0].min() if (grouped["mean"] > 0).any() else grouped["mean"].min()
mean_ratio = highest_mean_val / lowest_mean_val if lowest_mean_val != 0 else float("inf")

print(f"\nGroup with highest mean: '{highest_mean_group}' (mean = {highest_mean_val:.4f})")
print(f"Group with highest std:  '{highest_std_group}' (std = {grouped['std'].max():.4f})")
print(f"\nRatio of highest to lowest group mean: {mean_ratio:.2f}")

if mean_ratio > 2:
    print(f"  → Ratio ({mean_ratio:.2f}) > 2: suggests '{cat_col}' carries predictive signal.")
else:
    print(f"  → Ratio ({mean_ratio:.2f}) ≤ 2: the categorical feature may have limited predictive power.")

print(f"\n[README NOTE] The high within-group standard deviation in '{highest_std_group}'")
print(f"means that knowing the month alone is insufficient to predict the burned area")
print(f"reliably for fires in that month — additional features are needed.")

# =============================================================================
# TASK 10 — Save cleaned dataset
# =============================================================================
print(SEPARATOR + "TASK 10: Save Cleaned Dataset" + SEPARATOR)

df.to_csv("cleaned_data.csv", index=False)
print(f"Cleaned dataset saved to 'cleaned_data.csv'")
print(f"Final shape: {df.shape}")
print(f"Final dtypes:\n{df.dtypes}")
print(f"Remaining nulls: {df.isnull().sum().sum()}")

print(SEPARATOR + "ANALYSIS COMPLETE — All tasks executed successfully." + SEPARATOR)
