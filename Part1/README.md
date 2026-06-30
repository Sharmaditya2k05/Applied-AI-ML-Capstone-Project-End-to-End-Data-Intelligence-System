# Forest Fires — Part 1: Data Acquisition, Cleaning, and Exploratory Analysis

## 1. Dataset Description

This project uses the **Forest Fires** dataset from the UCI Machine Learning Repository, originally compiled by Paulo Cortez and Aníbal Morais (University of Minho, 2007). The dataset records **517 fire incidents** in the Montesinho Natural Park (northeast Portugal) across **13 variables**:

| Variable | Type        | Description                                             | Range          |
|----------|-------------|---------------------------------------------------------|----------------|
| X        | Integer     | x-axis spatial coordinate within the park map           | 1–9            |
| Y        | Integer     | y-axis spatial coordinate within the park map           | 2–9            |
| month    | Categorical | Month of the year (jan–dec)                             | 12 levels      |
| day      | Categorical | Day of the week (mon–sun)                               | 7 levels       |
| FFMC     | Float       | Fine Fuel Moisture Code (FWI system)                    | 18.7–96.20     |
| DMC      | Float       | Duff Moisture Code (FWI system)                         | 1.1–291.3      |
| DC       | Float       | Drought Code (FWI system)                               | 7.9–860.6      |
| ISI      | Float       | Initial Spread Index (FWI system)                       | 0.0–56.10      |
| temp     | Float       | Temperature in Celsius                                  | 2.2–33.30      |
| RH       | Integer     | Relative humidity in %                                  | 15–100         |
| wind     | Float       | Wind speed in km/h                                      | 0.40–9.40      |
| rain     | Float       | Outside rain in mm/m²                                   | 0.0–6.4        |
| **area** | Float       | **Burned area of the forest in hectares (target)**      | 0.00–1090.84   |

**Citation:** P. Cortez and A. Morais. *A Data Mining Approach to Predict Forest Fires using Meteorological Data.* Proceedings of the 13th EPIA 2007, pp. 512–523.

## Dataset Choice

**Dataset:** Forest Fires Dataset
**Source:** UCI Machine Learning Repository

### Justification

This project uses the Forest Fires Dataset, which contains meteorological data and Fire Weather Index (FWI) system components collected for forest fire incidents in the Montesinho Natural Park, Portugal.

This dataset was selected because it satisfies all assignment requirements:

- 517 records (above the required minimum of 500 rows).
- More than 5 numerical features, including temperature, relative humidity, wind speed, rain, FFMC, DMC, DC, and ISI.
- At least 2 categorical features, including month and day of the week.
- A continuous target variable (Burned Area) suitable for regression.
- The same target can be converted into a binary classification problem by splitting it into Above Median and Below Median burned area classes.
- Contains complex, skewed distributions and natural outliers, making it highly suitable for exploratory data analysis, outlier handling, feature engineering, traditional machine learning, ensemble methods, and LLM-powered prediction explanations.

For these reasons, this dataset provides a realistic end-to-end machine learning workflow while meeting every requirement of the assignment.

---

## 2. Null Value Analysis and Imputation Justification

### Null Analysis Results

All 13 columns have **0% null values**. No column exceeds the 20% null threshold. The dataset arrived complete from the source.

*Note: Although the dataset contains no missing values, the required median-imputation logic was implemented to demonstrate the preprocessing strategy that would be applied if missing values were present.*

### Why Median Over Mean?

The **median** was chosen as the imputation statistic rather than the mean for the following reasons:

1. **Robustness to outliers:** The median is not affected by extreme values. In this dataset, `area` has a maximum of 1,090.84 ha while the median is only 0.54 ha — the mean (12.89) is ~24× the median, showing how extreme fires pull the mean far from the "typical" observation.

2. **Robustness to skewed distributions:** Several columns are heavily skewed (rain: skew = 19.74, area: skew = 12.80). In a positively skewed distribution, the mean is dragged rightward by the long tail of high values, making it unrepresentative of where most data points actually lie.

3. **Preserving the distribution shape:** Imputing with the median adds values at the center of the distribution rather than inflating the tail influence, which is especially important when the downstream model may be sensitive to the distribution of imputed values.

---

## 3. Duplicate Detection

- **4 duplicate rows** were detected and removed.
- Shape changed from (517, 13) → (513, 13).
- Duplicate removal did **not** change any column's null percentage (all remained at 0%).

---

## 4. Data Type Corrections

*Note: Inspection showed that no numeric column was incorrectly inferred as object by pandas. Therefore, no numeric dtype correction was necessary. The required dtype optimization was demonstrated by converting the repetitive string columns (`month` and `day`) from `object` to `category`.*

| Column | Before   | After    | Rationale                                    |
|--------|----------|----------|----------------------------------------------|
| month  | object   | category | Repetitive string with only 12 unique values |
| day    | object   | category | Repetitive string with only 7 unique values  |

- **Memory before conversion:** 106,832 bytes (104.3 KB)
- **Memory after conversion:** 48,038 bytes (46.9 KB)
- **Memory saved:** 58,794 bytes (**55.0% reduction**)

Converting repetitive strings to `category` dtype dramatically reduces memory because pandas stores a small integer lookup table instead of duplicating full string objects.

---

## 5. Skewness Analysis

### Skewness Values (sorted by |skew|)

| Column | Skewness | Direction |
|--------|----------|-----------|
| rain   | 19.7395  | Positive  |
| area   | 12.8022  | Positive  |
| FFMC   | −6.5489  | Negative  |
| ISI    | 2.5282   | Positive  |
| DC     | −1.1114  | Negative  |
| RH     | 0.8544   | Positive  |
| wind   | 0.5812   | Positive  |
| DMC    | 0.5455   | Positive  |
| Y      | 0.4158   | Positive  |
| temp   | −0.3292  | Negative  |
| X      | 0.0260   | Positive  |

### Column with Highest Absolute Skewness: `rain` (skew = 19.74)

**Interpretation:** The `rain` column has extreme **positive skewness**, meaning the vast majority of observations have zero or near-zero rainfall, with a very long right tail created by a few rare rainy events (max = 6.4 mm/m²). The distribution is heavily concentrated at zero.

**Consequence for mean imputation:** In a positively skewed distribution, the mean is pulled *upward* by the extreme high values in the right tail. For `rain`, the mean (0.022) is far above the median (0.000). If we were to impute missing `rain` values with the mean, we would systematically overestimate rainfall for the majority of observations where the true value is likely zero. The **median** is the correct choice because it represents the most common data pattern (no rain).

---

## 6. Outlier Detection with IQR

### `area` (Burned Area)

| Statistic      | Value     |
|----------------|-----------|
| Q1             | 0.00      |
| Q3             | 6.57      |
| IQR            | 6.57      |
| Lower bound    | −9.86     |
| Upper bound    | 16.43     |
| Outlier count  | 62 (12.09%) |
| Outlier range  | 17.20 – 1,090.84 |

**Interpretation:** 12% of observations are IQR outliers, all on the upper end. This is expected because forest fire burned area is inherently right-skewed — most fires are small, but a few catastrophic fires burn vastly more area. These are **genuine extreme events**, not measurement errors.

### `ISI` (Initial Spread Index)

| Statistic      | Value     |
|----------------|-----------|
| Q1             | 6.40      |
| Q3             | 11.00     |
| IQR            | 4.60      |
| Lower bound    | −0.50     |
| Upper bound    | 17.90     |
| Outlier count  | 14 (2.73%)  |
| Outlier range  | 18.00 – 56.10 |

**Interpretation:** 2.7% of ISI values exceed the upper bound. ISI measures the expected rate of fire spread — extreme values correspond to very dry, windy conditions where fires spread rapidly. Again, these are real meteorological conditions, not errors.

### Outlier Handling Decision

**Outliers will be retained** for modeling in Part 2. Rationale:
- They represent real, physically meaningful fire events and weather conditions.
- Dropping them would remove the very observations the model needs to learn from to predict large fires.
- Instead, a **log transform** (`ln(area + 1)`) will be applied to the target variable `area` to compress the scale of extreme values, reducing their disproportionate influence on loss functions without discarding data.
- For `ISI`, no transformation is needed — the outliers are moderate and naturally bounded by meteorological limits.

---

## 7. Visualization Interpretations

### 7a. Line Plot — Temperature by Row Index
Shows the temperature variation across observations in their recorded order. Temperature fluctuates between roughly 2°C and 33°C, with visible clusters of higher temperatures corresponding to summer months (August–September), which dominate the dataset.

### 7b. Bar Chart — Mean Burned Area by Month
Reveals that **May** and **September** have the highest average burned areas, while **January** and **November** have the lowest (near zero). This aligns with the expectation that late spring and summer/early fall conditions (heat, dryness) promote larger fires. Note that May and November have very few observations (2 and 1, respectively), so their means should be interpreted cautiously.

### 7c. Histogram — Distribution of `rain`
The histogram of `rain` (skew = 19.74) shows an **extreme right-skewed distribution** — essentially a spike at zero with a barely visible tail. The vast majority of fire observations occurred during dry conditions (rain = 0). This confirms that rainfall during fire events is exceptionally rare in this dataset.

### 7d. Scatter Plot — Temperature vs. Relative Humidity
The scatter plot shows a **moderate negative correlation** (Pearson r ≈ −0.53). As temperature increases, relative humidity tends to decrease. This is physically expected: warmer air can hold more moisture, so at a given absolute humidity, relative humidity drops as temperature rises. The relationship is approximately linear with moderate scatter, suggesting other factors (wind, time of day) also influence humidity.

### 7e. Box Plot — Temperature by Month
The box plot reveals clear **seasonal temperature patterns**:
- Summer months (June–September) have the highest medians (~20–25°C) and tight interquartile ranges.
- Winter/spring months (February–April) show lower medians (~10–15°C) with wider spreads.
- August has the most observations and a relatively compact distribution around 20–25°C.
- December shows high variability despite fewer observations.

---

## 8. Correlation Heat Map Analysis

### Strongest Correlation
The pair with the highest absolute Pearson correlation is **DMC ↔ DC** (r = 0.6817).

**Does this imply causation?** No. Both DMC (Duff Moisture Code) and DC (Drought Code) are components of the Canadian Forest Fire Weather Index (FWI) system. They both measure aspects of fuel moisture — DMC tracks moisture in the medium organic layer (duff), while DC tracks deep, compact organic layers. Their correlation arises because:

1. **Common driver:** Both are influenced by the same weather variables — temperature, relative humidity, and rainfall — accumulated over different time windows.
2. **Seasonal covariation:** Both increase during prolonged dry, warm periods (summer) and decrease after sustained rain, meaning seasonal weather patterns drive both upward or downward simultaneously.
3. **Shared physical process:** Drying of organic material at different soil depths follows similar patterns driven by evapotranspiration.

**Plausible alternative explanation:** The variable **temperature** (`temp`) is a common causal driver of both DMC and DC. Higher temperatures increase evaporation rates, drying out both the medium and deep organic layers simultaneously. This confounding variable likely explains much of the DMC–DC correlation.

---

## 9a. Imputation Strategy Comparison

### Two Highest-Skewness Columns

| Column | Mean    | Median | Skewness | Chosen Statistic |
|--------|---------|--------|----------|------------------|
| rain   | 0.0218  | 0.0000 | +19.74   | **Median**       |
| area   | 12.8916 | 0.5400 | +12.80   | **Median**       |

### Justification

- **`rain` (skew = +19.74, positive):** The mean (0.022) is pulled *upward* by rare high-rainfall events. The median (0.000) correctly reflects that the overwhelming majority of fires occur with no rainfall. Imputing with the mean would introduce artificial non-zero rain values where zeros are expected.

- **`area` (skew = +12.80, positive):** The mean (12.89) is pulled *upward* by a few catastrophic fires (max = 1,090.84 ha). The median (0.54) represents the typical small fire much more faithfully. Imputing with the mean would systematically overestimate missing burned areas.

**General principle:** For **positively skewed** columns, the mean is inflated above the central mass of the data by extreme high values in the right tail. For **negatively skewed** columns, the mean is depressed below the central mass by extreme low values in the left tail. In both cases, the **median** is the more representative measure of central tendency and should be preferred for imputation.

After applying median imputation, `isnull().sum()` confirms **0 nulls remain** in both columns.

---

## 9b. Spearman vs. Pearson Correlation

### Top 3 Column Pairs with Largest |Spearman − Pearson| Difference

| Pair            | Pearson | Spearman | \|Difference\| |
|-----------------|---------|----------|---------------|
| FFMC ↔ ISI      | 0.5321  | 0.7851   | **0.2530**    |
| DC ↔ temp       | 0.4976  | 0.3071   | **0.1905**    |
| FFMC ↔ temp     | 0.4316  | 0.5959   | **0.1643**    |

### Interpretation

1. **FFMC ↔ ISI** (|Spearman| > |Pearson|): The relationship is **monotonic but non-linear**. Both are FWI components: FFMC measures fine fuel moisture and ISI measures expected fire spread rate. They move together consistently (higher FFMC → higher ISI), but not proportionally — the relationship likely saturates at high FFMC values where small moisture changes produce large ISI responses. The Spearman correlation (0.785) much better captures this rank-consistent pattern than Pearson (0.532).

2. **DC ↔ temp** (|Pearson| > |Spearman|): The relationship appears **approximately linear** when measured by Pearson (0.498), but the Spearman correlation (0.307) is notably lower. This paradoxical result suggests the presence of influential outliers or non-monotonic patterns in certain temperature ranges that disrupt rank consistency while linear regression captures the overall trend. The high Pearson value may be partially inflated by a few extreme observations.

3. **FFMC ↔ temp** (|Spearman| > |Pearson|): The relationship is **monotonic but non-linear**. Higher temperatures promote faster drying of fine fuels, increasing FFMC, but this effect saturates as FFMC approaches its upper bound (~96). The rank-based Spearman (0.596) outperforms Pearson (0.432), confirming the non-proportional nature.

### Feature Selection Guidance for Part 2

**Spearman correlation** will be the primary guide for feature selection because:
- Multiple variable pairs show strong monotonic but non-linear relationships that Pearson underestimates.
- Fire behavior involves threshold effects and non-proportional responses (e.g., fine fuels dry rapidly once temperature exceeds a threshold, then saturate).
- Spearman is also robust to the extreme skewness in `rain` and `area`, which can distort Pearson correlations.

Pearson will be used as a secondary check, particularly for pairs like DC ↔ temp where it detects a stronger signal.

---

## 9c. Grouped Aggregation — Burned Area by Month

| Month | Mean (ha) | Std (ha) | Count |
|-------|-----------|----------|-------|
| jan   | 0.00      | 0.00     | 2     |
| feb   | 6.28      | 12.34    | 20    |
| mar   | 3.90      | 8.58     | 53    |
| apr   | 8.89      | 19.93    | 9     |
| may   | 19.24     | 27.21    | 2     |
| jun   | 6.21      | 17.37    | 16    |
| jul   | 14.37     | 50.85    | 32    |
| aug   | 12.63     | 60.68    | 182   |
| sep   | 17.94     | 87.65    | 172   |
| oct   | 6.64      | 13.70    | 15    |
| nov   | 0.00      | NaN      | 1     |
| dec   | 13.33     | 6.61     | 9     |

### Key Findings

- **Highest mean group:** `may` (mean = 19.24 ha) — though based on only 2 observations, so unreliable.
- **Highest standard deviation group:** `sep` (std = 87.65 ha) — September fires range from zero to massive (max = 1,090.84 ha in the full dataset).
- **Mean ratio (highest / lowest non-zero):** 19.24 / 3.90 = **4.94×**

### Modeling Implications

1. **High within-group variance concern:** The extremely high standard deviation in September (87.65 ha vs. a mean of 17.94 ha, a coefficient of variation of ~489%) means that **knowing the month is September alone is far from sufficient to predict burned area** for fires in that month. September fires are heterogeneous — they include both zero-area and catastrophic fires. Additional features (weather conditions, FWI indices) are essential for reliable predictions within high-variance months.

2. **Predictive signal:** The mean ratio of 4.94× (highest to lowest group mean) exceeds 2×, suggesting that **month does carry meaningful predictive signal** — different months have substantially different average fire severities. This feature should be retained in the model, likely with one-hot or ordinal encoding.

---

## 10. Output Files

| File              | Description                                |
|-------------------|--------------------------------------------|
| `analysis.py`     | Complete analysis script (runs top-to-bottom) |
| `cleaned_data.csv`| Cleaned dataset (513 rows, 13 columns, 0 nulls) |
| `plot_1_lineplot.png` | Line plot: Temperature by row index     |
| `plot_2_barchart.png` | Bar chart: Mean area by month           |
| `plot_3_histogram.png`| Histogram: Rain distribution (most skewed) |
| `plot_4_scatterplot.png`| Scatter: Temperature vs. Relative Humidity |
| `plot_5_boxplot.png`  | Box plot: Temperature by month          |
| `plot_6_correlation_heatmap.png` | Pearson correlation heatmap   |
