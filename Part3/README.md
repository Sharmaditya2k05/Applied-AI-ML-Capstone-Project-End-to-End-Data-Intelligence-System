# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

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

## 1. Decision Tree Baseline (Unconstrained)

| Metric           | Value  |
|------------------|--------|
| Training accuracy| 0.9927 |
| Test accuracy    | 0.6214 |
| Train-test gap   | 0.3713 |
| Actual tree depth| 17     |

### Overfitting Analysis

The unconstrained tree shows **severe overfitting**: it achieves 99.3% training accuracy but only 62.1% test accuracy — a gap of 37 percentage points. The tree memorizes the training data by growing to depth 17 (creating hyper-specific rules for individual samples) but fails to generalize.

### Why Decision Trees Are High-Variance Models

Decision trees are **greedy, recursive partitioners**: at each node, they choose the single best split for the current data without considering how that split affects future nodes. Once a split is made, it is never revisited or revised. This greedy strategy means:
- Small changes in the training data can lead to completely different splits at the root, cascading into entirely different tree structures.
- Deep trees create very specific decision boundaries that fit the noise in the training set, not just the signal.
- This instability (high variance) is the fundamental motivation for ensemble methods like Random Forests that average over many such trees.

---

## 2. Controlled Decision Tree

| Metric           | Unconstrained (depth=17) | Controlled (depth=5, mss=20) |
|------------------|--------------------------|------------------------------|
| Training accuracy| 0.9927                   | 0.6561                       |
| Test accuracy    | 0.6214                   | 0.5825                       |
| Train-test gap   | 0.3713                   | 0.0736                       |

### Role of Hyperparameters

- **`max_depth=5`**: Limits how deep the tree can grow. A shallower tree captures only the strongest, most general patterns in the data, **reducing variance** (less overfitting) at the cost of some **bias** (may miss subtle patterns). The controlled tree's train-test gap shrank from 0.37 to 0.07 — evidence of much less overfitting.

- **`min_samples_split=20`**: Prevents a node from splitting if it contains fewer than 20 samples. This avoids creating splits that respond to noise in small subsets — a split on 5 samples might perfectly separate them, but that rule is almost certainly spurious. Requiring 20 samples ensures each split has enough statistical support to be meaningful.

The controlled tree sacrifices some training accuracy (65.6% vs 99.3%) but achieves a more honest picture of its generalization ability — the test accuracy (58.3%) is much closer to its training accuracy.

---

## 3. Gini vs Entropy Comparison

| Criterion | Test Accuracy |
|-----------|---------------|
| Gini      | 0.5631        |
| Entropy   | 0.5340        |

### Formulas

**Gini Impurity:**

$$\text{Gini}(t) = 1 - \sum_{i=1}^{C} p_i^2$$

**Entropy:**

$$\text{Entropy}(t) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

where $p_i$ is the proportion of class $i$ samples at node $t$, and $C$ is the number of classes.

**What Gini = 0 means:** A node has Gini impurity of 0 when all samples in it belong to a single class (a "pure" node). For binary classification, if $p_1 = 1.0$ and $p_0 = 0.0$, then Gini = $1 - (1.0^2 + 0.0^2) = 0$. This is the ideal split outcome — perfect class separation.

In practice, Gini and Entropy produce very similar trees. Gini is slightly faster to compute (no logarithm) and tends to isolate the most frequent class in its own branch, while Entropy produces slightly more balanced splits. On this dataset, Gini marginally outperforms Entropy (56.3% vs 53.4%).

---

## 4. Random Forest

| Metric           | Value  |
|------------------|--------|
| Training accuracy| 0.9805 |
| Test accuracy    | 0.5340 |
| ROC-AUC          | 0.5811 |

*Note: Scaling is unnecessary for tree-based models (Decision Trees, Random Forests, Gradient Boosting), but it is reused here for consistency with Part 2 and because the same train/test feature matrices were required.*

### Top 5 Features by Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1    | temp    | 0.1304     |
| 2    | RH      | 0.1223     |
| 3    | DMC     | 0.1050     |
| 4    | wind    | 0.1048     |
| 5    | FFMC    | 0.0969     |

### How Random Forest Computes Feature Importance

Random Forest feature importance is computed as the **average reduction in Gini impurity** achieved by all splits on that feature, averaged across all trees in the ensemble. Each time a feature is used to split a node, the weighted Gini impurity decreases; this decrease is summed across all splits and all trees, then normalized to sum to 1.0.

This differs fundamentally from linear regression coefficients:
- **Linear regression coefficients** measure the marginal change in the predicted value for a one-unit change in the feature, assuming a linear additive relationship.
- **Random Forest importances** measure how useful a feature is for *splitting* data into purer groups, regardless of the functional form of the relationship. A feature can have high RF importance even with a non-linear, threshold-based effect that linear regression would miss.

### Bagging (Bootstrap Aggregation) Concept

Random Forest uses **bagging** to reduce variance: each of the 100 trees is trained on a **bootstrap sample** — a random sample with replacement from the training data, equal in size to the original training set. This means each tree sees about 63% of unique training points (some appear multiple times, others not at all). Additionally, at each split, only a random subset of ⌊√17⌋ = 4 features is considered (not all 17). This **double randomization** (random data × random features) ensures the trees are diverse — they make different errors on different subsets. When the predictions of all 100 trees are averaged (or majority-voted), the individual errors tend to cancel out, producing an ensemble that is **much more stable (lower variance)** than any single deep decision tree, while maintaining similar or lower bias.

---

## 4a. Gradient Boosting

| Metric           | Value  |
|------------------|--------|
| Training accuracy| 0.8976 |
| Test accuracy    | 0.5922 |
| ROC-AUC          | 0.6004 |

Gradient Boosting achieves the highest test-set AUC (0.6004) among the base models, suggesting that its sequential error-correction approach captures patterns that bagging alone misses.

---

## 4b. Feature Ablation Study

### 5 Lowest-Importance Features Removed

| Feature  | Importance |
|----------|------------|
| day_sun  | 0.0111     |
| day_thu  | 0.0111     |
| day_tue  | 0.0109     |
| day_mon  | 0.0100     |
| rain     | 0.0034     |

### Results

| Model                     | Test AUC |
|---------------------------|----------|
| RF — All 17 features      | 0.5811   |
| RF — Reduced 12 features  | 0.6114   |
| AUC difference            | +0.0303  |

### Interpretation

The reduced model **outperformed** the full model by +0.03 AUC, indicating the 5 removed features (4 day-of-week dummies and `rain`) were **genuinely uninformative** and were adding noise to the model. The day-of-week features have very low importance because the day a fire occurs has no meaningful physical relationship to its size. The `rain` feature is nearly constant at zero (only 8 non-zero values in the entire dataset), providing no useful discriminative signal.

### Production Trade-Off

Deploying the simpler 12-feature model in production offers several advantages:
- **Lower inference cost:** Fewer features to collect, validate, and feed to the model.
- **Lower maintenance burden:** Five fewer data pipelines to maintain, fewer potential points of failure.
- **Better generalization:** Removing noisy features reduces overfitting risk.
- **Acceptable only if AUC degradation is below threshold:** In this case, AUC actually *improved*, making the reduced model strictly superior. As a general principle, if removing features causes AUC to drop by more than ~1–2%, the features likely carry real signal and should be retained despite the added complexity.

---

## 5. Cross-Validated Comparison (5-Fold Stratified)

| Model                   | Mean AUC | Std AUC |
|-------------------------|----------|---------|
| Logistic Regression     | 0.4347   | 0.0451  |
| Decision Tree (depth=5) | 0.5051   | 0.0543  |
| Random Forest           | 0.5450   | 0.0295  |
| Gradient Boosting       | 0.5373   | 0.0531  |

### Why Cross-Validation Is More Reliable

A single train-test split evaluates the model on one specific subset of 103 test samples, which may be unrepresentative. Cross-validation trains and evaluates the model on **5 different train-test partitions**, using every sample for testing exactly once. This provides:
1. **A mean performance estimate** that is less sensitive to the particular random split.
2. **A standard deviation** that quantifies how much performance varies across splits, measuring the model's stability.
3. **Detection of overfitting:** A model that scores well on one split but poorly on others (high std) is unreliable.

The Random Forest has the highest mean CV AUC (0.5450) **and** the lowest standard deviation (0.0295), making it the most robust model.

---

## 6. Hyperparameter Tuning with GridSearchCV

### Best Parameters

| Parameter            | Best Value |
|----------------------|------------|
| n_estimators         | 200        |
| max_depth            | None       |
| min_samples_leaf     | 5          |

**Best CV AUC:** 0.5476
**Test-set AUC:** 0.6167

### Grid Search Statistics

- **Total configurations evaluated:** 18 (3 × 3 × 2 = n_estimators × max_depth × min_samples_leaf)
- **Total model fits:** 90 (18 configurations × 5 folds)

### Grid Search vs Randomized Search Trade-Off

**Grid Search** exhaustively evaluates every combination in the parameter grid, guaranteeing the best configuration within the grid is found. However, its cost grows exponentially with the number of hyperparameters (curse of dimensionality). With 18 configurations × 5 folds = 90 fits, this is manageable.

**Randomized Search** samples a fixed number of random configurations from the parameter space, making it far more efficient when the grid is large (e.g., 6+ hyperparameters). It may miss the absolute best configuration but often finds a near-optimal one in a fraction of the time. For this dataset with only 3 hyperparameters, Grid Search is computationally feasible and guarantees completeness.

---

## 6a. Manual Learning Curve

| Training Fraction | Samples | Training AUC | Test AUC |
|-------------------|---------|-------------|----------|
| 20%               | 82      | 0.9787      | 0.5133   |
| 40%               | 164     | 0.9666      | 0.5057   |
| 60%               | 246     | 0.9756      | 0.5909   |
| 80%               | 328     | 0.9714      | 0.5989   |
| 100%              | 410     | 0.9672      | 0.6167   |

### Interpretation

1. **Training AUC decreases slightly** as the training set grows (0.979 → 0.967). This is expected for high-variance models — with more data, the model cannot memorize every sample as perfectly, so training performance drops slightly. This confirms the Random Forest is fitting the training data very closely.

2. **Test AUC increases** with more training data (0.513 → 0.617), and it is **still rising at 100%**. This means the model benefits from additional training examples — each new sample helps the ensemble learn more robust patterns.

3. **Conclusion: The model is DATA-LIMITED.** The test AUC has not plateaued — it rose from 0.599 (80%) to 0.617 (100%), a meaningful gain. This implies that **collecting more fire observations would likely improve model performance further.** The model's capacity (200 trees, unlimited depth) is not the bottleneck; the limiting factor is the relatively small dataset (410 training samples). This is consistent with the Forest Fires dataset's known difficulty — 517 total observations is modest for a complex, noisy regression/classification task.

*Note on testing: Although the instructions mention testing on `X_test_scaled`, the tuned GridSearch pipeline includes its own `StandardScaler()`. Therefore, the unscaled `X_test` is passed to `predict_proba()` to avoid double scaling.*

---

## 7. Model Serialization

- `best_model.pkl` saved to disk using `joblib.dump()`.
- Reload-and-predict demo confirmed working:
  - Sample 1 (hot/dry August Saturday): predicted **large fire** (P = 0.5470)
  - Sample 2 (cool/humid February Monday): predicted **small fire** (P = 0.4098)

---

## 8. Summary Comparison Table (All Models, Parts 2 + 3)

| Model                     | CV Mean AUC | CV Std AUC | Test AUC |
|---------------------------|-------------|------------|----------|
| Logistic Regression       | 0.4347      | 0.0451     | 0.5973   |
| Decision Tree (depth=5)   | 0.5051      | 0.0543     | 0.5881   |
| Random Forest             | 0.5450      | 0.0295     | 0.5811   |
| Gradient Boosting         | 0.5373      | 0.0531     | 0.6004   |
| **GridSearch Best Pipeline** | **0.5476** | **0.0604** | **0.6167** |

### Final Recommendation

The **GridSearch-tuned Random Forest pipeline** (n_estimators=200, max_depth=None, min_samples_leaf=5) is the recommended model for the client. It achieves the highest test-set AUC (0.6167) and the highest cross-validated AUC (0.5476), while being packaged in a reproducible `sklearn.Pipeline` that handles imputation and scaling automatically. Although all models struggle on this dataset — reflecting the inherent difficulty of predicting fire size from weather data alone — the tuned Random Forest captures non-linear interactions between features (temp, RH, DMC, wind) that linear models miss. Its `min_samples_leaf=5` constraint prevents overfitting to small subsets, and the learning curve analysis confirms that collecting more data would further improve performance.

---

## Output Files

| File               | Description                                              |
|--------------------|----------------------------------------------------------|
| `analysis.py`      | Complete Part 3 script (runs top-to-bottom)              |
| `cleaned_data.csv` | Cleaned dataset from Part 1                              |
| `forestfires.csv`  | Original raw dataset                                     |
| `best_model.pkl`   | Serialized best pipeline (GridSearchCV result)           |
