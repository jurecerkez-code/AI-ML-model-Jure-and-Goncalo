# 🍄 Mushroom Classification — ML Pipeline

A complete end-to-end machine learning pipeline for classifying mushrooms as **edible** or **poisonous** using the UCI Mushroom Dataset. Includes exploratory data analysis, multi-model comparison, detailed evaluation, and categorical correlation analysis via **Cramér's V**.

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `mushroomsML.ipynb` | Main notebook — full ML pipeline (EDA → training → evaluation → Cramér's V heatmap) |
| `mushrooms_100k.csv` | Extended dataset — 100,000 rows, 23 columns (synthetic expansion of UCI original) |

---

## 🗂️ Dataset

`mushrooms_100k.csv` contains **100,000 mushroom samples** across **23 categorical features**:

| Column | Description |
|--------|-------------|
| `class` | Target — `e` (edible) or `p` (poisonous) |
| `cap-shape` | Bell, conical, convex, flat, knobbed, sunken |
| `cap-surface` | Fibrous, grooves, scaly, smooth |
| `cap-color` | 10 colour categories |
| `odor` | Almond, anise, creosote, fishy, foul, musty, none, pungent, spicy |
| `gill-color` | 12 colour categories |
| `stalk-root` | Bulbous, club, cup, equal, rhizomorphs, rooted — contains `?` missing values (~30.5%) |
| *(+ 16 more features)* | See notebook Cell 3 for full cardinality breakdown |

**Class balance:** ~52.5% edible / ~47.5% poisonous

---

## 🔬 Pipeline Overview

The notebook (`mushroomsML.ipynb`) follows a structured 12-step pipeline:

```
1.  Imports
2.  Load & Explore Data
3.  Exploratory Data Analysis (EDA)
4.  Data Cleaning & Preprocessing
5.  Train / Test Split
6.  Model Training & Comparison
7.  Select Best Model
8.  Detailed Evaluation
9.  Visualisations
10. Experiment Log
11. Cramer's V Correlation Table
12. Cramer's V Heatmap
```

### Preprocessing

- Missing values (`?`) in `stalk-root` replaced with the **mode**
- `veil-type` dropped — only 1 unique value, carries no predictive information
- All 21 remaining categorical features **one-hot encoded** via `ColumnTransformer`
- **Stratified 80/20 train/test split** (`random_state=42`)

### Models Compared

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear model (`max_iter=1000`) |
| Decision Tree | Interpretable, no ensemble |
| Random Forest | 100 estimators |
| Gradient Boosting | 100 estimators |
| XGBoost | 100 estimators, `eval_metric=logloss` |

All models are evaluated with **5-fold stratified cross-validation** on the training set, followed by a final **test accuracy + ROC-AUC** score on the held-out 20%.

### Evaluation Metrics

- Cross-validation accuracy (mean ± std)
- Test set accuracy
- ROC-AUC
- Full classification report (precision, recall, F1 per class)
- Confusion matrix

**Accuracy target:** >= 97.9% ✅

---

## 📊 Visualisations

**Cell 9** produces a 2x2 figure (`mushroom_results.png`) for the best model:

1. **Model Comparison** — horizontal bar chart of test accuracy across all 5 models
2. **Confusion Matrix** — edible vs poisonous prediction breakdown
3. **Top 15 Feature Importances** — tree-based models only; N/A for Logistic Regression
4. **ROC Curve** — false positive rate vs true positive rate with AUC annotation

### Cramer's V Heatmap (Cells 11 & 12)

Because every feature is categorical, standard Pearson correlation is not applicable. This notebook uses **Cramer's V** — a symmetric measure of association between categorical variables:

```
V = sqrt( chi2 / (n * min(r-1, c-1)) )
```

| Strength | Cramer's V Range |
|----------|-----------------|
| 🔴 Strong | >= 0.5 |
| 🟡 Moderate | 0.3 – 0.49 |
| ⚪ Weak | < 0.3 |

- **Cell 11** prints a ranked table of every feature's association with the `class` target
- **Cell 12** renders a full **feature x feature heatmap** — annotated with values, coloured `YlOrRd`, range 0–1 — making it easy to spot clusters of correlated features at a glance

---

## 🚀 Getting Started

### Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
scipy
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy
```

### Run

1. Clone the repo and ensure `mushrooms_100k.csv` is in the **same directory** as `mushroomsML.ipynb`
2. Open the notebook:

```bash
jupyter notebook mushroomsML.ipynb
```

3. Run all cells top-to-bottom (`Kernel → Restart & Run All`)

> **Note:** Cell 2 checks for an alternative CSV filename (`mushrooms_100k (2).csv`). If your file is named `mushrooms_100k.csv`, this check will print a "does not exist" message — that is expected and safe to ignore.

---

## 📋 Experiment Log (Summary)

| Item | Value |
|------|-------|
| Dataset | UCI Mushroom extended (100k rows) |
| Task | Binary classification — edible vs poisonous |
| Raw features | 22 categorical columns |
| Features after encoding | ~100+ one-hot columns |
| Train / test split | 80% / 20%, stratified |
| Cross-validation | 5-fold stratified KFold |
| Models tested | 5 |
| Best model | Determined at runtime (typically Random Forest or XGBoost) |
| Accuracy target | >= 97.9% |

---

## 📚 References

- [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom) — original 8,124-row source
- [Cramer's V — Wikipedia](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V)
- [scikit-learn documentation](https://scikit-learn.org/)
- [XGBoost documentation](https://xgboost.readthedocs.io/)

---

*Built for the mushroom classification hackathon 🍄*
