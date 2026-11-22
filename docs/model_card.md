# Model card: income classifier

| | |
|---|---|
| **Model** | Gradient boosting (scikit-learn `HistGradientBoostingClassifier`, 150 iterations, learning rate 0.1), after a `ColumnTransformer` that prepares the features |
| **Version** | 1.0.1, trained on 2026-11-22 with seed 42 (the same model and numbers as 1.0.0) |
| **Task** | Predict whether a person's yearly income is above 50,000 US dollars |
| **Output** | A probability between 0 and 1. At 0.5 or more the answer is "above 50K" |
| **File** | `models/income_model.joblib`, created by `python scripts/train.py` (not stored in the repository) |

## Intended use

- A teaching and portfolio example of a complete tabular prediction workflow.
- A model for other example projects to load, for instance to show how a model is served.

## Not intended for

- Any decision about a real person: hiring, lending, pricing, benefits, or anything similar.
- Estimating incomes today, or outside the United States. The data is from 1994.

## Training data

The [Adult dataset](https://archive.ics.uci.edu/dataset/2/adult) (UCI Machine Learning Repository, CC BY 4.0), taken from the 1994 United States census. After cleaning: 48,790 people, 23.9% of them above 50K. 39,032 rows were used for training and cross-validation, and 9,758 were kept for the test.

**Inputs:** age, years of education, hours worked per week, capital gain, capital loss, work class, marital status, occupation, relationship, race, sex, native country.

**Left out:** `fnlwgt` (a census sampling weight) and `education` (the same information as the years of education).

## Results on the test rows

| F1 | ROC AUC | RMSE | MAE | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|
| 0.724 | 0.933 | 0.292 | 0.171 | 0.876 | 0.776 | 0.678 |

A baseline that always predicts the average scores F1 0.000, ROC AUC 0.500, RMSE 0.427, MAE 0.364 and accuracy 0.761. What each metric means, and its weakness, is in the [README](../README.md#what-each-metric-says-and-where-it-is-weak).

## Known limits and risks

- **Unequal errors between groups.** The model misses 43% of high-earning women and 30% of high-earning men, and most high earners among the young and those with less schooling. See the [error analysis](error_analysis.md).
- **Sensitive inputs.** Sex, race and native country are inputs, and the model reproduces the income patterns of 1994.
- **Old data.** Incomes, jobs and the meaning of 50,000 dollars have changed since 1994.
- **Not tuned, not calibrated.** The settings are near the defaults, the cut-off is 0.5, and the probabilities were not checked against observed rates.
- **Short CPU run.** The model was trained in under a minute on a laptop. A longer search would likely do better.

## How to reproduce

`python scripts/train.py` writes the same `reports/metrics.json` on every run with the same settings. The settings are in `src/tabular_prediction/config.py`.

Checked on Windows with Python 3.13 and the library versions pinned in `pyproject.toml`. On another system or with other library versions, the last decimals may differ.
