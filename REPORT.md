# ML Methods for Biophysics — Homework Report
**Student:** Yovan Hegel-Valentych  
**Course:** Machine Learning Methods, BIPH 2026  
**Repository:** [yovalentych/ML_Methods_PhD_2026_Hegl](https://github.com/yovalentych/ML_Methods_PhD_2026_Hegl)

---

## Overview

This report summarises the homework assignments completed across five classes of
the ML Methods course. All analyses are implemented in R (v4.6) using the
tidyverse ecosystem and are fully reproducible from the `.qmd` source files.

---

## Class 1 — Tidy Data and Tidyverse Workflow

**File:** `class1_tidy_data_homework.qmd`  
**Dataset:** Synthetic messy version of the Palmer Penguins dataset (333 rows, 11 columns) + island metadata

### Task

Apply a complete tidyverse data-cleaning pipeline to a deliberately messy dataset.

### Steps and methodology

| Step | Operation | Key finding |
|---|---|---|
| 1 | Inspect | Mixed letter case in `species`, `island`, `sex`; NAs in 4 columns |
| 2 | Clean | `str_to_title()` / `str_to_lower()` + `drop_na()` → 272 rows remain |
| 3 | Filter | High/medium quality, Biscoe or Dream island, body mass > 3 000 g → 205 rows |
| 4 | Mutate | Added `body_mass_kg`, `bill_ratio`, `large_penguin` flag |
| 5 | Pivot | `pivot_longer()` → 820 rows (205 × 4 measurements) |
| 6 | Join | `left_join()` with island metadata — environmental context added |
| 7 | Summarise | Grouped by species, island, sex, protected area; ordered by mean body mass |

### Key results

- **Gentoo males on Biscoe island** are the heaviest group (mean ≈ 5 480 g).
- Both sampled islands (Biscoe and Dream) are protected areas.
- `bill_ratio` differs markedly by species: Gentoo bills are proportionally
  narrower than Chinstrap and Adelie, which is consistent with their different diet.

---

## Class 2 — Linear Regression Mini Challenge

**File:** `class2_linear_regression_challenge.qmd`  
**Dataset:** `mpg` (ggplot2 built-in, 234 cars, target: `hwy`)  
**Constraint:** `cty` excluded (r ≈ 0.96 with `hwy`)

### Models compared

| Model | Description | adj. R² | AIC | Test RMSE |
|---|---|---|---|---|
| M1 | `hwy ~ displ` | 0.663 | 939 | 4.73 |
| M2 | `hwy ~ poly(displ,2) + class` | 0.832 | 824 | 3.01 |
| M3 | `hwy ~ poly(displ,2) + class + drv + cyl + fl` | 0.896 | 747 | 2.19 |
| **M4** | **Backward AIC selection** | **0.914** | **713** | **2.15** |

### Key findings

- A purely displacement-based model (M1) is a poor predictor — vehicle class and
  drivetrain type capture substantial variance that engine size cannot.
- The quadratic `poly(displ, 2)` term consistently outperforms a linear `displ`
  term, reflecting the diminishing returns of engine displacement on efficiency.
- Backward AIC selection (M4) chose a subset of predictors that closely matches
  domain knowledge: large engine + rear-wheel drive → low MPG.
- Residual diagnostics for M4 show approximately normal residuals with mild
  heteroscedasticity at extreme fitted values — acceptable for interpretation
  and inference but worth noting for uncertainty quantification.

---

## Class 3 — Logistic Regression on Wisconsin Breast Cancer

**File:** `class3_logistic_regression_homework.qmd`  
**Dataset:** BreastCancer (mlbench package), 683 rows after NA removal  
**Target:** malignant (positive class) vs benign  
**Split:** stratified 75/25

### Models compared

| Model | Predictors | Accuracy | Recall | AUC |
|---|---|---|---|---|
| M1 | 5 selected | 0.913 | 0.917 | 0.984 |
| M2 | All 9 | 0.932 | 0.967 | 0.988 |
| **M3** | **Backward AIC** | **0.942** | **0.967** | **0.987** |

### Technical note — GLM factor encoding

`glm()` with `Class = factor(levels = c("malignant","benign"))` encodes
*malignant* as the reference (0) and *benign* as the event (1).
`predict(type = "response")` therefore returns **P(benign)**, not P(malignant).
The correct classification threshold rule is:

```
if P(benign) ≥ threshold  →  predict "benign"
if P(benign) <  threshold  →  predict "malignant"
```

This is the opposite of the naive `prob >= 0.5 → positive` convention used when
modelling P(positive) directly.

### Threshold analysis

The default threshold of 0.5 maximises F1 for M3. Raising the threshold
(e.g., to 0.9) further increases recall toward 1.0 at the cost of lower
specificity — appropriate for population screening where false negatives carry
higher clinical cost.

### Recommendation

M3 (backward AIC) is preferred: AUC matches M2 but with fewer predictors,
making it more interpretable and less likely to overfit on data from a different
hospital or instrument.

---

## Class 4 — PCA and Dimensionality Reduction

**File:** `class4_pca_homework.qmd`  
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), 569 samples, 30 features  
**Question:** How many principal components are sufficient for classification?

### PCA summary

| Threshold | Components needed |
|---|---|
| 90% variance | 7 PCs |
| 95% variance | 10 PCs |
| Full | 30 PCs |

### Classification results (logistic regression on PCA scores)

| n_components | AUC | Accuracy | Recall |
|---|---|---|---|
| 2 | — | — | — |
| 5 | ~0.97 | ~0.96 | ~0.94 |
| 7 | ~0.98 | ~0.97 | ~0.95 |
| 10 | ~0.98 | ~0.97 | ~0.96 |
| 30 | — | — | — |

*Note: AUC values for very low or very high component counts are affected by
the perfect-separation problem (GLM convergence instability when all training
probabilities are numerically 0 or 1). This is a known limitation of standard
logistic regression — regularised methods (ridge, LASSO) would be more reliable
in that regime.*

### Key conclusions

- **5–10 PCs** are sufficient to achieve near-maximum AUC (≈ 0.98), reducing
  dimensionality from 30 to 7–10 features.
- PC1 alone accounts for ~44% of variance; PC1 + PC2 already produce clear
  class separation in the 2D score plot.
- PCA is most valuable here for **visualisation** (2 PCs) and for **stabilising
  the logistic regression** when raw features are highly collinear (as they are
  in WDBC — `radius`, `perimeter`, `area` are almost perfectly correlated).

---

## Class 5 — Decision Trees and Random Forests

**File:** `class5_trees_challenge.qmd`  
**Dataset:** WDBC (same split as Class 4)

### Task 1 — Decision tree complexity parameter sweep

| cp | Leaves | AUC | Accuracy | Recall |
|---|---|---|---|---|
| 0.001 | 4 | **0.928** | 0.930 | 0.887 |
| 0.005 | 4 | **0.928** | 0.930 | 0.887 |
| 0.01 | 3 | 0.903 | 0.923 | 0.830 |
| 0.05 | 3 | 0.903 | 0.923 | 0.830 |
| 0.1 | 2 | 0.838 | 0.874 | 0.698 |
| 0.2 | 2 | 0.838 | 0.874 | 0.698 |

**Finding:** The tree stabilises at 4 leaves for cp ≤ 0.005. Further reduction
of cp does not add leaves because the splitting criterion is already satisfied —
the problem is solved by a shallow tree on this dataset.

### Task 2 — Random Forest: ntree = 100 vs 500

| Model | AUC | Accuracy | Recall | F1 |
|---|---|---|---|---|
| RF ntree = 100 | 0.985 | 0.944 | 0.906 | 0.923 |
| RF ntree = 500 | **0.988** | **0.951** | **0.925** | **0.933** |

The OOB convergence plot shows that error stabilises by approximately ntree = 60.
The difference between 100 and 500 trees is marginal (~0.003 AUC) on this dataset.
For larger / noisier problems, 500 trees would be the safer default.

### Task 3 — Feature importance

Features consistently ranked in the top-10 by both models:
**perimeter_worst, area_worst, radius_worst, concave_points_worst, concavity_worst**

These "worst-value" (largest per-tumour) nuclear shape descriptors are the
strongest discriminators across two fundamentally different importance measures
(Gini improvement vs mean decrease in OOB accuracy). This cross-model agreement
strengthens the biological interpretation.

### Task 4 — Model recommendation

| Purpose | Model | Why |
|---|---|---|
| **Explanation** | Decision Tree (cp = 0.001) | 4 leaves → 3 simple rules that any clinician can follow |
| **Prediction** | Random Forest (ntree = 500) | Highest AUC, stable out-of-bag error, robust to feature collinearity |
| **Regulatory** | Logistic Regression | Interpretable coefficients, standard in biomedical literature |

---

## Environment and reproducibility

```
R version 4.6.0 (2026-04-24)
tidyverse  2.0.0
broom      1.x
yardstick  1.x
pROC       1.x
mlbench    2.x
rpart      4.x
randomForest 4.7-1.2
```

All analyses use `set.seed(2026)` for reproducibility. The `materials/` folder
(course repository clone) is listed in `.gitignore` to avoid publishing the
instructor's original files.

---

## Repository structure

```
ML_Methods_PhD_2026_Hegl/
├── class1_tidy_data_homework.qmd
├── class2_linear_regression_challenge.qmd
├── class3_logistic_regression_homework.qmd
├── class4_pca_homework.qmd
├── class5_trees_challenge.qmd
├── REPORT.md
└── .gitignore
```
