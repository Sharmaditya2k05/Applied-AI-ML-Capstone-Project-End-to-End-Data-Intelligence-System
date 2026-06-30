"""
Part 2 — Supervised Machine Learning: Build, Train, and Evaluate
Forest Fires Dataset — Regression & Classification
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score,
)
from imblearn.over_sampling import SMOTE

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

SEP = "\n" + "=" * 80 + "\n"
np.random.seed(42)

# =============================================================================
# TASK 1 — Load data & define labels
# =============================================================================
print(SEP + "TASK 1: Load Data & Define Labels" + SEP)

df = pd.read_csv("cleaned_data.csv")
print(f"Loaded cleaned_data.csv: {df.shape}")

# Target column
target_col = "area"

# Regression label — continuous burned area (log-transformed to handle skewness)
y_reg = np.log1p(df[target_col])
print(f"\nRegression label (y_reg): ln({target_col} + 1) — log-transformed burned area")
print(f"  mean = {y_reg.mean():.4f}, median = {y_reg.median():.4f}, std = {y_reg.std():.4f}")

# Classification label — binarize at median
median_val = y_reg.median()
y_clf = (y_reg > median_val).astype(int)
print(f"\nClassification label (y_clf): area > {median_val} → 1 (above-median burned area), else 0 (below-or-equal-median burned area)")
print(f"  Class distribution:\n{y_clf.value_counts().to_string()}")
print(f"  Class 0: {(y_clf == 0).sum()} ({(y_clf == 0).mean()*100:.1f}%)")
print(f"  Class 1: {(y_clf == 1).sum()} ({(y_clf == 1).mean()*100:.1f}%)")

# Feature matrix — all columns except target
X = df.drop(columns=[target_col])
print(f"\nFeature matrix X shape: {X.shape}")

# =============================================================================
# TASK 2 — Encode categorical columns
# =============================================================================
print(SEP + "TASK 2: Encode Categorical Columns" + SEP)

# MONTH has a natural cyclical order → Label encode (Jan=1 … Dec=12)
month_order = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
               "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
X["month"] = X["month"].map(month_order)
print("Label-encoded 'month' (natural calendar order: jan=1 → dec=12)")
print(f"  Unique values: {sorted(X['month'].unique())}")

# DAY has no meaningful natural order → One-hot encode, drop first
print("\nOne-hot encoding 'day' (no natural order: mon, tue, ... are nominal)")
X = pd.get_dummies(X, columns=["day"], drop_first=True, dtype=int)
print(f"  Created columns: {[c for c in X.columns if c.startswith('day_')]}")
print(f"  drop_first=True avoids multicollinearity (k-1 dummies for k categories)")

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Features: {list(X.columns)}")

# =============================================================================
# TASK 3 — Leak-free train-test split and scaling
# =============================================================================
print(SEP + "TASK 3: Train-Test Split & Scaling" + SEP)

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
# Use same split indices for classification label
y_clf_train = y_clf.loc[X_train.index]
y_clf_test = y_clf.loc[X_test.index]

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# Fit scaler ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStandardScaler fit on TRAINING data only (leak-free).")
print("[README NOTE] Fitting the scaler on the full dataset would leak test-set")
print("statistics (mean, std) into the training process, biasing evaluation metrics.")

# =============================================================================
# TASK 4 — Regression: Linear Regression + Ridge
# =============================================================================
print(SEP + "TASK 4: Regression — Linear Regression & Ridge" + SEP)

# --- 4a. Ordinary Least Squares ---
lr = LinearRegression()
lr.fit(X_train_scaled, y_reg_train)
y_pred_reg = lr.predict(X_test_scaled)

mse_ols = mean_squared_error(y_reg_test, y_pred_reg)
r2_ols = r2_score(y_reg_test, y_pred_reg)

print("=== OLS Linear Regression ===")
print(f"  MSE  = {mse_ols:.4f}")
print(f"  R²   = {r2_ols:.4f}")

# Coefficients
feature_names = X.columns.tolist()
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": lr.coef_
}).sort_values("Coefficient", key=abs, ascending=False)

print(f"\n  Intercept = {lr.intercept_:.4f}")
print("\n  Coefficients (sorted by |value|):")
print(coef_df.to_string(index=False))

top3_features = coef_df.head(3)
print(f"\n  Top 3 features by |coefficient|:")
for _, row in top3_features.iterrows():
    direction = "positive" if row["Coefficient"] > 0 else "negative"
    print(f"    {row['Feature']}: {row['Coefficient']:.4f} ({direction})")

# --- 4b. Ridge Regression ---
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)

print("\n=== Ridge Regression (alpha=1.0) ===")
print(f"  MSE  = {mse_ridge:.4f}")
print(f"  R²   = {r2_ridge:.4f}")

# Comparison table
print("\n--- OLS vs Ridge Comparison ---")
print(f"{'Model':<25} {'MSE':>12} {'R²':>10}")
print("-" * 48)
print(f"{'Linear Regression (OLS)':<25} {mse_ols:>12.4f} {r2_ols:>10.4f}")
print(f"{'Ridge (alpha=1.0)':<25} {mse_ridge:>12.4f} {r2_ridge:>10.4f}")

# Ridge coefficients
ridge_coef_df = pd.DataFrame({
    "Feature": feature_names,
    "OLS Coef": lr.coef_,
    "Ridge Coef": ridge.coef_,
    "|Difference|": np.abs(lr.coef_ - ridge.coef_)
}).sort_values("|Difference|", ascending=False)
print("\n  Coefficient comparison (OLS vs Ridge):")
print(ridge_coef_df.to_string(index=False))

# =============================================================================
# TASK 5 — Classification: Logistic Regression
# =============================================================================
print(SEP + "TASK 5: Classification — Logistic Regression" + SEP)

# --- 5a. Check class imbalance ---
print("Class distribution in TRAINING set (before resampling):")
train_counts = y_clf_train.value_counts()
print(train_counts.to_string())
minority_pct = train_counts.min() / train_counts.sum() * 100
print(f"Minority class: {minority_pct:.1f}%")

# Apply SMOTE if minority < 35%, otherwise use class_weight='balanced'
if minority_pct < 35:
    print("\n⚠ Minority class < 35% → Applying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_clf_train_resampled = smote.fit_resample(
        X_train_scaled, y_clf_train
    )
    print(f"After SMOTE: {pd.Series(y_clf_train_resampled).value_counts().to_string()}")
    used_smote = True
else:
    print("\nMinority class ≥ 35% → Using class_weight='balanced' instead of SMOTE.")
    X_train_resampled = X_train_scaled
    y_clf_train_resampled = y_clf_train
    used_smote = False

# --- 5b. Train Logistic Regression (C=1.0, baseline) ---
log_reg = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight=None if used_smote else "balanced"
)
log_reg.fit(X_train_resampled, y_clf_train_resampled)

y_pred_clf = log_reg.predict(X_test_scaled)
y_proba_clf = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression (C=1.0, baseline) ===")

# Confusion matrix
cm = confusion_matrix(y_clf_test, y_pred_clf)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]:>4}  FP={cm[0,1]:>4}")
print(f"  FN={cm[1,0]:>4}  TP={cm[1,1]:>4}")

# Classification report
print(f"\nClassification Report:")
print(classification_report(y_clf_test, y_pred_clf, digits=4))

# Accuracy, Precision, Recall, F1
accuracy = (y_pred_clf == y_clf_test.values).mean()
precision = precision_score(y_clf_test, y_pred_clf)
recall = recall_score(y_clf_test, y_pred_clf)
f1 = f1_score(y_clf_test, y_pred_clf)
auc_baseline = roc_auc_score(y_clf_test, y_proba_clf)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc_baseline:.4f}")

# --- 5c. ROC Curve ---
fpr, tpr, thresholds_roc = roc_curve(y_clf_test, y_proba_clf)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color="#E91E63", linewidth=2, label=f"Logistic Regression (AUC = {auc_baseline:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.1, color="#E91E63")
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curve — Logistic Regression (C=1.0)", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=12)
ax.annotate(f"AUC = {auc_baseline:.3f}", xy=(0.55, 0.35), fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)
plt.savefig("plot_roc_curve.png")
plt.close()
print("\n✓ ROC curve saved: plot_roc_curve.png")

# =============================================================================
# TASK 5b — Decision-threshold sensitivity
# =============================================================================
print(SEP + "TASK 5b: Decision-Threshold Sensitivity" + SEP)

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
print("-" * 48)

threshold_results = []
for t in thresholds:
    y_pred_t = (y_proba_clf >= t).astype(int)
    p = precision_score(y_clf_test, y_pred_t, zero_division=0)
    r = recall_score(y_clf_test, y_pred_t, zero_division=0)
    f = f1_score(y_clf_test, y_pred_t, zero_division=0)
    threshold_results.append({"Threshold": t, "Precision": p, "Recall": r, "F1": f})
    print(f"{t:>10.2f} | {p:>10.4f} | {r:>10.4f} | {f:>10.4f}")

best_threshold = max(threshold_results, key=lambda x: x["F1"])
print(f"\nThreshold that maximises F1: {best_threshold['Threshold']:.2f} "
      f"(F1 = {best_threshold['F1']:.4f})")

print("\n[README NOTE] Formulas:")
print("  Precision = TP / (TP + FP)")
print("  Recall    = TP / (TP + FN)")
print("  Lowering threshold → higher recall but lower precision (more false positives)")
print("  Raising threshold → higher precision but lower recall (more false negatives)")

# =============================================================================
# TASK 6 — Regularization experiment (C=0.01)
# =============================================================================
print(SEP + "TASK 6: Regularization Experiment (C=0.01)" + SEP)

log_reg_reg = LogisticRegression(
    C=0.01,
    max_iter=1000,
    random_state=42,
    class_weight=None if used_smote else "balanced"
)
log_reg_reg.fit(X_train_resampled, y_clf_train_resampled)

y_pred_reg_clf = log_reg_reg.predict(X_test_scaled)
y_proba_reg_clf = log_reg_reg.predict_proba(X_test_scaled)[:, 1]

precision_reg = precision_score(y_clf_test, y_pred_reg_clf)
recall_reg = recall_score(y_clf_test, y_pred_reg_clf)
f1_reg = f1_score(y_clf_test, y_pred_reg_clf)
auc_reg = roc_auc_score(y_clf_test, y_proba_reg_clf)

print(f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 72)
print(f"{'LogReg C=1.0 (baseline)':<30} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {auc_baseline:>10.4f}")
print(f"{'LogReg C=0.01 (regularized)':<30} {precision_reg:>10.4f} {recall_reg:>10.4f} {f1_reg:>10.4f} {auc_reg:>10.4f}")

if auc_baseline > auc_reg:
    print(f"\n→ Reducing C from 1.0 to 0.01 WORSENED AUC ({auc_baseline:.4f} → {auc_reg:.4f}).")
else:
    print(f"\n→ Reducing C from 1.0 to 0.01 IMPROVED AUC ({auc_baseline:.4f} → {auc_reg:.4f}).")

# =============================================================================
# TASK 7 — Bootstrap confidence interval for AUC difference
# =============================================================================
print(SEP + "TASK 7: Bootstrap Confidence Interval for AUC Difference" + SEP)

n_bootstrap = 500
auc_diffs = []

y_test_arr = y_clf_test.values

for i in range(n_bootstrap):
    # Sample indices with replacement
    idx = np.random.choice(len(y_test_arr), size=len(y_test_arr), replace=True)
    y_boot = y_test_arr[idx]

    # Skip if only one class in bootstrap sample
    if len(np.unique(y_boot)) < 2:
        continue

    proba_c1_boot = y_proba_clf[idx]
    proba_c001_boot = y_proba_reg_clf[idx]

    auc_c1 = roc_auc_score(y_boot, proba_c1_boot)
    auc_c001 = roc_auc_score(y_boot, proba_c001_boot)
    auc_diffs.append(auc_c1 - auc_c001)

auc_diffs = np.array(auc_diffs)
mean_diff = auc_diffs.mean()
ci_lower = np.percentile(auc_diffs, 2.5)
ci_upper = np.percentile(auc_diffs, 97.5)

print(f"Bootstrap samples used: {len(auc_diffs)} / {n_bootstrap}")
print(f"\nAUC difference (C=1.0 minus C=0.01):")
print(f"  Mean difference:   {mean_diff:.4f}")
print(f"  95% CI:            [{ci_lower:.4f}, {ci_upper:.4f}]")

if ci_lower > 0:
    print(f"\n→ The 95% CI EXCLUDES zero → C=1.0 model's advantage is statistically reliable.")
elif ci_upper < 0:
    print(f"\n→ The 95% CI EXCLUDES zero (negative) → C=0.01 model is reliably better.")
else:
    print(f"\n→ The 95% CI INCLUDES zero → the AUC difference is NOT statistically reliable.")
    print(f"  The two models may perform comparably; the observed difference could be due to chance.")

print(SEP + "PART 2 COMPLETE — All tasks executed successfully." + SEP)
