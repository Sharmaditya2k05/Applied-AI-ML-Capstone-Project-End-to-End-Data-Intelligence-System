"""
Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline
Forest Fires Dataset
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

SEP = "\n" + "=" * 80 + "\n"
np.random.seed(42)

# =============================================================================
# DATA PREPARATION (same as Part 2)
# =============================================================================
print(SEP + "DATA PREPARATION (reproducing Part 2 split)" + SEP)

df = pd.read_csv("cleaned_data.csv")
target_col = "area"

y_reg = df[target_col].copy()
y_clf = (y_reg > y_reg.median()).astype(int)

X = df.drop(columns=[target_col])

# Encode month (ordinal) and day (one-hot)
month_order = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
               "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
X["month"] = X["month"].map(month_order)
X = pd.get_dummies(X, columns=["day"], drop_first=True, dtype=int)

# Same train-test split as Part 2
X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# Scale (leak-free)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

feature_names = X.columns.tolist()
print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
print(f"Features: {len(feature_names)}")

# =============================================================================
# TASK 1 — Decision Tree baseline (unconstrained)
# =============================================================================
print(SEP + "TASK 1: Decision Tree Baseline (Unconstrained)" + SEP)

dt_unconstrained = DecisionTreeClassifier(random_state=42)
dt_unconstrained.fit(X_train_scaled, y_clf_train)

train_acc_unc = accuracy_score(y_clf_train, dt_unconstrained.predict(X_train_scaled))
test_acc_unc = accuracy_score(y_clf_test, dt_unconstrained.predict(X_test_scaled))

print(f"Unconstrained Decision Tree (max_depth=None):")
print(f"  Training accuracy: {train_acc_unc:.4f}")
print(f"  Test accuracy:     {test_acc_unc:.4f}")
print(f"  Train-test gap:    {train_acc_unc - test_acc_unc:.4f}")
print(f"  Actual tree depth: {dt_unconstrained.get_depth()}")

if train_acc_unc - test_acc_unc > 0.10:
    print("  ⚠ OVERFITTING: large gap between train and test accuracy.")
else:
    print("  Train-test gap is moderate.")

# =============================================================================
# TASK 2 — Controlled Decision Tree
# =============================================================================
print(SEP + "TASK 2: Controlled Decision Tree (max_depth=5, min_samples_split=20)" + SEP)

dt_controlled = DecisionTreeClassifier(
    max_depth=5, min_samples_split=20, random_state=42
)
dt_controlled.fit(X_train_scaled, y_clf_train)

train_acc_ctrl = accuracy_score(y_clf_train, dt_controlled.predict(X_train_scaled))
test_acc_ctrl = accuracy_score(y_clf_test, dt_controlled.predict(X_test_scaled))

print(f"Controlled Decision Tree:")
print(f"  Training accuracy: {train_acc_ctrl:.4f}")
print(f"  Test accuracy:     {test_acc_ctrl:.4f}")
print(f"  Train-test gap:    {train_acc_ctrl - test_acc_ctrl:.4f}")
print(f"  Actual tree depth: {dt_controlled.get_depth()}")

print(f"\n--- Comparison ---")
print(f"{'Model':<35} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>10}")
print("-" * 66)
print(f"{'Unconstrained (depth='+str(dt_unconstrained.get_depth())+')':<35} "
      f"{train_acc_unc:>10.4f} {test_acc_unc:>10.4f} {train_acc_unc - test_acc_unc:>10.4f}")
print(f"{'Controlled (depth=5, mss=20)':<35} "
      f"{train_acc_ctrl:>10.4f} {test_acc_ctrl:>10.4f} {train_acc_ctrl - test_acc_ctrl:>10.4f}")

# =============================================================================
# TASK 3 — Gini vs Entropy comparison
# =============================================================================
print(SEP + "TASK 3: Gini vs Entropy Comparison (max_depth=5)" + SEP)

dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
dt_gini.fit(X_train_scaled, y_clf_train)
test_acc_gini = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))

dt_entropy = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)
dt_entropy.fit(X_train_scaled, y_clf_train)
test_acc_entropy = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))

print(f"{'Criterion':<12} {'Test Accuracy':>14}")
print("-" * 28)
print(f"{'Gini':<12} {test_acc_gini:>14.4f}")
print(f"{'Entropy':<12} {test_acc_entropy:>14.4f}")

print("\n[README NOTE] Gini impurity = 1 - Σ(pi²)")
print("             Entropy       = -Σ(pi × log₂(pi))")
print("             Gini = 0 means all samples in the node belong to one class (pure node).")

# =============================================================================
# TASK 4 — Random Forest
# =============================================================================
print(SEP + "TASK 4: Random Forest" + SEP)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_clf_train)

train_acc_rf = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
test_acc_rf = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
auc_rf = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])

print(f"Random Forest (n_estimators=100, max_depth=10):")
print(f"  Training accuracy: {train_acc_rf:.4f}")
print(f"  Test accuracy:     {test_acc_rf:.4f}")
print(f"  ROC-AUC:           {auc_rf:.4f}")

# Feature importances
importances = rf.feature_importances_
feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(f"\nTop 5 Features by Importance:")
print(f"{'Rank':<6} {'Feature':<15} {'Importance':>12}")
print("-" * 34)
for i, (_, row) in enumerate(feat_imp.head(5).iterrows(), 1):
    print(f"{i:<6} {row['Feature']:<15} {row['Importance']:>12.4f}")

print(f"\nAll feature importances:")
print(feat_imp.to_string(index=False))

# =============================================================================
# TASK 4a — Gradient Boosting
# =============================================================================
print(SEP + "TASK 4a: Gradient Boosting" + SEP)

gb = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gb.fit(X_train_scaled, y_clf_train)

train_acc_gb = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
test_acc_gb = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
auc_gb = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])

print(f"Gradient Boosting (n_estimators=100, lr=0.1, max_depth=3):")
print(f"  Training accuracy: {train_acc_gb:.4f}")
print(f"  Test accuracy:     {test_acc_gb:.4f}")
print(f"  ROC-AUC:           {auc_gb:.4f}")

# =============================================================================
# TASK 4b — Feature Ablation Study
# =============================================================================
print(SEP + "TASK 4b: Feature Ablation Study" + SEP)

# Identify 5 lowest-importance features
lowest_5 = feat_imp.tail(5)["Feature"].tolist()
print(f"5 lowest-importance features: {lowest_5}")
print(f"Their importance scores:")
for _, row in feat_imp.tail(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Get column indices to keep
keep_cols = [i for i, f in enumerate(feature_names) if f not in lowest_5]
removed_cols = [i for i, f in enumerate(feature_names) if f in lowest_5]

X_train_reduced = X_train_scaled[:, keep_cols]
X_test_reduced = X_test_scaled[:, keep_cols]

print(f"\nReduced feature set: {len(keep_cols)} features (removed {len(removed_cols)})")

# Train reduced model with same hyperparameters
rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_reduced, y_clf_train)

auc_rf_full = auc_rf  # already computed
auc_rf_reduced = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])

print(f"\n{'Model':<30} {'Test AUC':>10}")
print("-" * 42)
print(f"{'RF — All features (17)':<30} {auc_rf_full:>10.4f}")
print(f"{'RF — Reduced features ('+str(len(keep_cols))+')':<30} {auc_rf_reduced:>10.4f}")
print(f"{'AUC difference':<30} {auc_rf_full - auc_rf_reduced:>+10.4f}")

if abs(auc_rf_full - auc_rf_reduced) < 0.02:
    print("\n→ AUC is similar: removed features were largely uninformative.")
elif auc_rf_reduced > auc_rf_full:
    print("\n→ AUC IMPROVED: removed features were adding noise.")
else:
    print("\n→ AUC dropped: removed features were contributing some signal.")

# =============================================================================
# TASK 5 — Cross-validated comparison
# =============================================================================
print(SEP + "TASK 5: Cross-Validated Comparison (5-Fold Stratified)" + SEP)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42,
                                               class_weight="balanced"),
    "Decision Tree (depth=5)": DecisionTreeClassifier(max_depth=5,
                                                       min_samples_split=20,
                                                       random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10,
                                             random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100,
                                                      learning_rate=0.1,
                                                      max_depth=3,
                                                      random_state=42),
}

cv_results = {}
print(f"{'Model':<30} {'Mean AUC':>10} {'Std AUC':>10}")
print("-" * 52)

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train,
                             cv=cv, scoring="roc_auc")
    cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
    print(f"{name:<30} {scores.mean():>10.4f} {scores.std():>10.4f}")

# =============================================================================
# TASK 6 — Hyperparameter Tuning with GridSearchCV
# =============================================================================
print(SEP + "TASK 6: GridSearchCV — Random Forest Pipeline" + SEP)

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}

grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=grid_cv, scoring="roc_auc",
    n_jobs=-1, return_train_score=True
)

# Use UNSCALED X_train (pipeline handles scaling)
grid_search.fit(X_train, y_clf_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV AUC:     {grid_search.best_score_:.4f}")

n_combos = 3 * 3 * 2  # n_estimators × max_depth × min_samples_leaf
n_fits = n_combos * 5  # × 5 folds
print(f"\nTotal configurations evaluated: {n_combos}")
print(f"Total model fits:              {n_fits} ({n_combos} configs × 5 folds)")

# Get best pipeline
best_pipeline = grid_search.best_estimator_

# Test-set AUC for best pipeline
y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]
auc_best = roc_auc_score(y_clf_test, y_proba_best)
print(f"Test-set AUC (best pipeline):  {auc_best:.4f}")

# =============================================================================
# TASK 6a — Manual Learning Curve
# =============================================================================
print(SEP + "TASK 6a: Manual Learning Curve" + SEP)

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

print(f"{'Training Fraction':>18} | {'Training AUC':>13} | {'Test AUC':>10}")
print("-" * 48)

learning_curve_data = []
for f in fractions:
    n_samples = int(f * len(X_train))
    X_sub = X_train.iloc[:n_samples]
    y_sub = y_clf_train.iloc[:n_samples]

    # Clone and fit the best pipeline
    from sklearn.base import clone
    pipe_clone = clone(best_pipeline)
    pipe_clone.fit(X_sub, y_sub)

    # Training AUC
    train_proba = pipe_clone.predict_proba(X_sub)[:, 1]
    train_auc = roc_auc_score(y_sub, train_proba)

    # Test AUC on fixed full test set.
    # [README NOTE] Although the instruction mentions X_test_scaled, the tuned
    # pipeline includes its own scaler, so the unscaled X_test is passed to avoid double scaling.
    test_proba = pipe_clone.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_clf_test, test_proba)

    learning_curve_data.append({
        "fraction": f, "n_samples": n_samples,
        "train_auc": train_auc, "test_auc": test_auc
    })
    print(f"{f:>15.0%} ({n_samples:>3}) | {train_auc:>13.4f} | {test_auc:>10.4f}")

# Check trend
test_aucs = [d["test_auc"] for d in learning_curve_data]
train_aucs = [d["train_auc"] for d in learning_curve_data]

print(f"\nTraining AUC trend: {train_aucs[0]:.4f} → {train_aucs[-1]:.4f} "
      f"({'decreasing' if train_aucs[-1] < train_aucs[0] else 'stable/increasing'})")
print(f"Test AUC trend:     {test_aucs[0]:.4f} → {test_aucs[-1]:.4f} "
      f"({'increasing' if test_aucs[-1] > test_aucs[0] else 'stable/decreasing'})")

if test_aucs[-1] > test_aucs[-2] + 0.005:
    print("→ Test AUC still rising at 100% → model is likely DATA-LIMITED.")
else:
    print("→ Test AUC has plateaued → model may be CAPACITY-LIMITED or the signal is weak.")

# =============================================================================
# TASK 7 — Serialize the best model
# =============================================================================
print(SEP + "TASK 7: Serialize Best Model" + SEP)

joblib.dump(best_pipeline, "best_model.pkl")
print("✓ Saved best pipeline to best_model.pkl")

# Reload and predict on 2 hand-crafted test rows
print("\n--- Reload and Predict Demo ---")
loaded_model = joblib.load("best_model.pkl")

# Create 2 sample rows with same feature columns as X
# Row 1: A hot, dry August Saturday fire scenario
# Row 2: A cool, humid February Monday scenario
sample_data = pd.DataFrame({
    "X": [7, 3],
    "Y": [5, 4],
    "month": [8, 2],          # August, February
    "FFMC": [92.5, 85.0],
    "DMC": [120.0, 30.0],
    "DC": [600.0, 100.0],
    "ISI": [12.0, 5.0],
    "temp": [28.0, 8.0],
    "RH": [25, 75],
    "wind": [4.0, 2.0],
    "rain": [0.0, 0.5],
    "day_mon": [0, 1],
    "day_sat": [1, 0],
    "day_sun": [0, 0],
    "day_thu": [0, 0],
    "day_tue": [0, 0],
    "day_wed": [0, 0],
})

predictions = loaded_model.predict(sample_data)
probabilities = loaded_model.predict_proba(sample_data)[:, 1]

print(f"Sample 1 (hot/dry August Saturday): class={predictions[0]}, P(large fire)={probabilities[0]:.4f}")
print(f"Sample 2 (cool/humid February Monday): class={predictions[1]}, P(large fire)={probabilities[1]:.4f}")
print("✓ Reload-and-predict runs without errors.")

# =============================================================================
# TASK 8 — Summary Comparison Table
# =============================================================================
print(SEP + "TASK 8: Summary Comparison Table" + SEP)

# Gather test-set AUC for all models
# Part 2 models — Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr_model.fit(X_train_scaled, y_clf_train)
auc_lr = roc_auc_score(y_clf_test, lr_model.predict_proba(X_test_scaled)[:, 1])

# Decision Tree controlled
auc_dt = roc_auc_score(y_clf_test, dt_controlled.predict_proba(X_test_scaled)[:, 1])

print(f"{'Model':<30} {'CV Mean AUC':>12} {'CV Std AUC':>12} {'Test AUC':>10}")
print("-" * 66)

summary_data = [
    ("Logistic Regression", cv_results["Logistic Regression"]["mean"],
     cv_results["Logistic Regression"]["std"], auc_lr),
    ("Decision Tree (depth=5)", cv_results["Decision Tree (depth=5)"]["mean"],
     cv_results["Decision Tree (depth=5)"]["std"], auc_dt),
    ("Random Forest", cv_results["Random Forest"]["mean"],
     cv_results["Random Forest"]["std"], auc_rf),
    ("Gradient Boosting", cv_results["Gradient Boosting"]["mean"],
     cv_results["Gradient Boosting"]["std"], auc_gb),
    ("GridSearch Best Pipeline", grid_search.best_score_,
     grid_search.cv_results_["std_test_score"][grid_search.best_index_], auc_best),
]

for name, cv_mean, cv_std, test_auc in summary_data:
    print(f"{name:<30} {cv_mean:>12.4f} {cv_std:>12.4f} {test_auc:>10.4f}")

# Identify best
best_model_name = max(summary_data, key=lambda x: x[1])
print(f"\n★ Recommended model: {best_model_name[0]}")
print(f"  CV AUC = {best_model_name[1]:.4f}, Test AUC = {best_model_name[3]:.4f}")

print(SEP + "PART 3 COMPLETE — All tasks executed successfully." + SEP)
