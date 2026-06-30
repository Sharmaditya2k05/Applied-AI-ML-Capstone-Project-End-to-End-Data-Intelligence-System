# Part 2 — Supervised Machine Learning: Build, Train, and Evaluate

## Dataset Choice

**Dataset:** Forest Fires Dataset
**Source:** UCI Machine Learning Repository

### Justification

This project uses the Forest Fires Dataset, which contains meteorological data and Fire Weather Index (FWI) system components collected for forest fire incidents in the Montesinho Natural Park, Portugal.

This dataset was selected because it satisfies all assignment requirements:

- 517 records (above the required minimum of 500 rows). After duplicate removal, the cleaned dataset used here contains 513 rows, satisfying the 500+ row requirement.
- More than 5 numerical features, including temperature, relative humidity, wind speed, rain, FFMC, DMC, DC, and ISI.
- At least 2 categorical features, including month and day of the week.
- A continuous target variable (Burned Area) suitable for regression.
- The same target can be converted into a binary classification problem by splitting it into Above Median and Below Median burned area classes.
- Contains complex, skewed distributions and natural outliers, making it highly suitable for exploratory data analysis, outlier handling, feature engineering, traditional machine learning, ensemble methods, and LLM-powered prediction explanations.

For these reasons, this dataset provides a realistic end-to-end machine learning workflow while meeting every requirement of the assignment.

---

## 1. Label Definitions

| Label   | Column | Definition                                                                   |
|---------|--------|-----------------------------------------------------------------------------|
| `y_reg` | area   | Continuous burned area, log-transformed (`ln(area+1)`) to handle skewness (regression target) |
| `y_clf` | area   | Binary: `1` if area > 0.54 ha (median), `0` otherwise (classification target) |

- **Regression:** Predict the log-transformed burned area (continuous).
- **Classification:** Predict whether a fire is "above-median burned area" (1) or "below-or-equal-median burned area" (0).

Class distribution after binarization: Class 0 = 257 (50.1%), Class 1 = 256 (49.9%) — nearly perfectly balanced.

---

## 2. Categorical Encoding

### `month` — Label Encoding (Ordinal)

Months have a **natural calendar order** (jan=1, feb=2, … dec=12) that reflects seasonal progression. This ordering is meaningful because fire behavior varies systematically with season — summer months have higher temperatures, lower humidity, and greater fire risk. Label encoding preserves this ordinal relationship, allowing the model to learn that `aug=8` is "between" `jul=7` and `sep=9` in terms of seasonal weather patterns.

### `day` — One-Hot Encoding (Nominal)

Days of the week have **no natural order** for fire prediction. There is no reason to believe that Wednesday is "between" Tuesday and Thursday in terms of fire risk. Label encoding days as integers (mon=1, tue=2, …) would impose a **false ordinal relationship**, causing the model to incorrectly interpret `sun=7` as "greater than" `mon=1`. One-hot encoding avoids this by creating independent binary indicator columns — one per day — with `drop_first=True` to prevent multicollinearity (the dropped category, `fri`, serves as the implicit reference level, and its information is encoded by all other dummy columns being 0).

**Result:** 12 original features → 17 features after encoding (11 numeric + 1 label-encoded month + 6 one-hot day dummies, with `day_fri` dropped).

---

## 3. Leak-Free Train-Test Split and Scaling

- **Split:** 80/20 train-test split using `train_test_split(test_size=0.2, random_state=42)`.
  - Train set: **410 samples**
  - Test set: **103 samples**

- **Scaling:** `StandardScaler` fit **only on training data**, then used to transform both train and test sets.

### Why Fitting the Scaler on Full Data Is Data Leakage

If we fit `StandardScaler` on the entire dataset (train + test), the scaler's mean and standard deviation parameters would encode statistical information from the test set. When we then use this scaler to transform training features, the training data effectively "sees" test-set patterns — the transformed feature values are computed using global statistics that include future/unseen data. This constitutes **data leakage** because:

1. The model trains on features that have been subtly biased toward the test set distribution.
2. Evaluation metrics become overly optimistic — they no longer measure true out-of-sample performance.
3. In production, the model would never have access to future data statistics at training time.

**Correct approach:** `scaler.fit(X_train)` only, then `scaler.transform(X_train)` and `scaler.transform(X_test)`.

---

## 4. Regression Model — Linear Regression & Ridge

### 4a. OLS Linear Regression Results

| Metric | Value    |
|--------|----------|
| MSE    | 519.7404 |
| R²     | −0.3496  |

The negative R² indicates the model performs **worse than a simple mean predictor**. This is expected because the `area` target is extremely skewed (most values near 0, a few very large fires), making linear regression a poor fit for this inherently non-linear, zero-inflated distribution.

### Top 3 Features by Absolute Coefficient

| Feature | Coefficient | Interpretation                                                  |
|---------|-------------|----------------------------------------------------------------|
| day_sat | +7.4092     | Fires on Saturday are associated with ~7.4 ha more burned area than the baseline (Friday), all else equal. |
| temp    | +6.8655     | A 1-standard-deviation increase in temperature is associated with ~6.9 ha more burned area. |
| X       | +5.6079     | A 1-standard-deviation increase in the X spatial coordinate (eastward) is associated with ~5.6 ha more area. |

**Interpreting coefficients on scaled features:**
- A **large positive coefficient** means that a 1-standard-deviation increase in that feature is associated with the coefficient's value increase in predicted burned area (ha).
- A **large negative coefficient** (e.g., DC = −5.24) means a 1-standard-deviation increase in that feature is associated with a decrease in predicted area by that many hectares.

### 4b. Ridge Regression (α = 1.0)

| Model                   | MSE      | R²      |
|-------------------------|----------|---------|
| Linear Regression (OLS) | 519.7404 | −0.3496 |
| Ridge (α = 1.0)         | 519.3970 | −0.3487 |

Ridge marginally improves both MSE and R², though neither model performs well on this target.

### Why Ridge Produces Different Coefficients

Ridge regression adds an **L2 penalty** (λ × Σβ²) to the OLS loss function, where the penalty strength is controlled by `alpha` (α = λ). This penalty discourages large coefficient values by shrinking them toward zero — features with less predictive power are penalized more heavily. The parameter `alpha` controls the trade-off: higher α applies stronger shrinkage, producing smaller and more uniform coefficients, while α = 0 recovers plain OLS. In this dataset, the difference is minimal (α = 1.0 is mild) because the features are already scaled and no single coefficient dominates excessively. Ridge is most beneficial when features are highly correlated (multicollinear) — it stabilizes coefficients that would otherwise be inflated and volatile under OLS.

---

## 5. Classification — Logistic Regression

### 5a. Class Imbalance Handling

| Class | Count | Percentage |
|-------|-------|------------|
| 0     | 202   | 49.3%      |
| 1     | 208   | 50.7%      |

The minority class (49.3%) exceeds 35%, so severe resampling (SMOTE) is unnecessary. Instead, **`class_weight='balanced'`** was used in the `LogisticRegression` constructor. This automatically adjusts the loss function to weight each class inversely proportional to its frequency, ensuring the model does not favor the majority class even in cases of mild imbalance.

### 5b. Evaluation Results (C = 1.0, threshold = 0.5)

**Confusion Matrix:**

|              | Predicted 0 | Predicted 1 |
|--------------|-------------|-------------|
| Actual 0     | TN = 34     | FP = 21     |
| Actual 1     | FN = 22     | TP = 26     |

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.5825 |
| Precision | 0.5532 |
| Recall    | 0.5417 |
| F1-score  | 0.5474 |
| AUC       | 0.5973 |

### 5c. Precision and Recall Formulas

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{26}{26 + 21} = 0.5532$$

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{26}{26 + 22} = 0.5417$$

- **Precision** answers: "Of all fires I predicted as large, what fraction actually were large?"
- **Recall** answers: "Of all fires that actually were large, what fraction did I correctly identify?"

### Which Metric Is More Important?

For forest fire prediction, **recall is more important** than precision. A false negative (failing to predict a large fire) is more costly than a false positive (over-predicting fire size):
- Missing a large fire → insufficient firefighting resources deployed → greater property damage, loss of life.
- Over-predicting fire size → extra resources mobilized → financial cost, but no safety risk.

The asymmetric cost structure makes recall the priority metric.

### AUC Interpretation

The AUC of **0.5973** means the model has a 59.7% probability of ranking a randomly chosen positive sample (large fire) higher than a randomly chosen negative sample (small fire). This is only modestly better than random guessing (AUC = 0.50), reflecting the inherent difficulty of predicting fire size from meteorological variables alone.

---

## 5d. Decision-Threshold Sensitivity

| Threshold | Precision | Recall | F1     |
|-----------|-----------|--------|--------|
| 0.30      | 0.4660    | 1.0000 | 0.6358 |
| 0.40      | 0.4891    | 0.9375 | 0.6429 |
| **0.50**  | 0.5532    | 0.5417 | 0.5474 |
| 0.60      | 0.7143    | 0.1042 | 0.1818 |
| 0.70      | 0.0000    | 0.0000 | 0.0000 |

### Key Findings

- **F1-maximising threshold: 0.40** (F1 = 0.6429) — lowering the threshold from the default 0.50 to 0.40 substantially improves F1 by boosting recall from 54% to 94% while only slightly reducing precision.

- **Which metric matters more?** As argued above, **recall** is more important for this fire-prediction task because the cost of missing a large fire (false negative) vastly exceeds the cost of a false alarm (false positive).

- **Threshold recommendation:** We would **lower the threshold** (toward 0.30–0.40) to optimise for recall. At threshold = 0.30, recall reaches 100% (all large fires detected), but precision drops to 46.6% — meaning roughly half of "large fire" predictions are false alarms. This is an acceptable trade-off in an emergency response context where over-deploying resources is far cheaper than under-deploying.

- **Cost of lowering the threshold:** More false positives → firefighting resources are mobilized unnecessarily for small fires, increasing operational costs and potentially diverting resources. The optimal operational threshold depends on the cost ratio of false negatives to false positives.

---

## 6. Regularization Experiment (C = 0.01)

| Model                        | Precision | Recall | F1     | AUC    |
|------------------------------|-----------|--------|--------|--------|
| LogReg C = 1.0 (baseline)    | 0.5532    | 0.5417 | 0.5474 | 0.5973 |
| LogReg C = 0.01 (regularized)| 0.5357    | 0.6250 | 0.5769 | 0.5936 |

### What C Controls

In `sklearn`'s `LogisticRegression`, `C` is the **inverse of regularization strength** (C = 1/λ). A smaller C means stronger L2 regularization — the model's coefficients are shrunk more aggressively toward zero, producing a simpler model that is less prone to overfitting but may underfit if the signal is subtle. Setting `C = 0.01` applies 100× stronger regularization than `C = 1.0`, heavily penalizing large coefficients. On this dataset, reducing C from 1.0 to 0.01 **slightly worsened AUC** (0.5973 → 0.5936), suggesting the baseline model was not overfitting and the stronger penalty removed useful signal. However, the C = 0.01 model shows improved recall (0.625 vs. 0.542), indicating that coefficient shrinkage redistributed predictions toward the positive class — a side effect of stronger regularization interacting with `class_weight='balanced'`.

---

## 7. Bootstrap Confidence Interval for AUC Difference

**Method:** 500 bootstrap samples drawn from the test set with replacement using `np.random.choice`. For each sample, the AUC difference (C=1.0 minus C=0.01) was computed.

| Statistic                    | Value   |
|------------------------------|---------|
| Mean AUC difference          | +0.0017 |
| 95% CI lower bound (2.5th)  | −0.0536 |
| 95% CI upper bound (97.5th) | +0.0691 |

### Interpretation

The 95% confidence interval **[−0.0536, +0.0691] includes zero**, meaning the observed AUC advantage of the C=1.0 model over the C=0.01 model is **not statistically reliable**. The difference could plausibly be zero or even negative (favoring C=0.01) under different random test-set compositions. The two regularization settings produce **comparable performance** on this dataset — neither is convincingly better. Given this finding, the simpler, more regularized model (C=0.01) is a reasonable choice under the parsimony principle, though the baseline C=1.0 remains the default recommendation given its marginally higher point estimate.

---

## Output Files

| File               | Description                                       |
|--------------------|---------------------------------------------------|
| `analysis.py`      | Complete Part 2 script (runs top-to-bottom)       |
| `cleaned_data.csv` | Cleaned dataset from Part 1                       |
| `forestfires.csv`  | Original raw dataset                              |
| `plot_roc_curve.png` | ROC curve with AUC annotation                  |
