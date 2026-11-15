# Metrics

Seed 42, 39,032 training rows, 9,758 test rows. Created by `python scripts/train.py`.

## Cross-validation (5 folds, mean ± standard deviation)

| Model | F1 score | ROC AUC | RMSE | MAE | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|
| baseline | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 | 0.4267 ± 0.0000 | 0.3642 ± 0.0000 | 0.7606 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| logistic_regression | 0.6430 ± 0.0090 | 0.8988 ± 0.0042 | 0.3267 ± 0.0035 | 0.2119 ± 0.0022 | 0.8444 ± 0.0038 | 0.7136 ± 0.0095 | 0.5851 ± 0.0088 |
| random_forest | 0.6651 ± 0.0093 | 0.9168 ± 0.0045 | 0.3103 ± 0.0034 | 0.2085 ± 0.0024 | 0.8620 ± 0.0034 | 0.7936 ± 0.0080 | 0.5724 ± 0.0103 |
| gradient_boosting **(chosen)** | 0.7100 ± 0.0094 | 0.9281 ± 0.0035 | 0.2971 ± 0.0039 | 0.1753 ± 0.0030 | 0.8715 ± 0.0041 | 0.7724 ± 0.0103 | 0.6570 ± 0.0103 |

## Test rows (used once, for the chosen model)

| Model | F1 score | ROC AUC | RMSE | MAE | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|
| gradient_boosting | 0.7237 | 0.9331 | 0.2919 | 0.1712 | 0.8761 | 0.7764 | 0.6777 |
| baseline | 0.0000 | 0.5000 | 0.4267 | 0.3642 | 0.7606 | 0.0000 | 0.0000 |

## Errors by group (chosen model, test rows)

| sex | Rows | Actually above 50K | Predicted above 50K | High incomes missed | False alarms | All errors |
|---|---|---|---|---|---|---|
| Female | 3,326 | 11.5% | 8.0% | 43.1% | 1.6% | 6.4% |
| Male | 6,432 | 30.4% | 27.6% | 30.1% | 9.1% | 15.5% |

| age_band | Rows | Actually above 50K | Predicted above 50K | High incomes missed | False alarms | All errors |
|---|---|---|---|---|---|---|
| under 25 | 1,663 | 1.5% | 0.5% | 72.0% | 0.1% | 1.1% |
| 25-34 | 2,463 | 15.6% | 12.4% | 40.3% | 3.6% | 9.3% |
| 35-44 | 2,446 | 34.3% | 30.2% | 30.8% | 9.8% | 17.0% |
| 45-54 | 1,783 | 38.0% | 36.5% | 24.9% | 12.8% | 17.4% |
| 55-64 | 958 | 32.6% | 28.1% | 35.9% | 10.7% | 18.9% |
| 65+ | 445 | 21.8% | 15.1% | 42.3% | 3.2% | 11.7% |

| education_band | Rows | Actually above 50K | Predicted above 50K | High incomes missed | False alarms | All errors |
|---|---|---|---|---|---|---|
| below high school | 1,292 | 5.8% | 1.9% | 69.3% | 0.1% | 4.1% |
| high school | 3,115 | 16.1% | 8.2% | 62.0% | 2.6% | 12.1% |
| some college | 2,909 | 18.6% | 16.5% | 39.2% | 6.3% | 12.4% |
| bachelor | 1,625 | 43.3% | 46.0% | 17.1% | 17.9% | 17.5% |
| above bachelor | 817 | 63.3% | 65.0% | 11.4% | 24.3% | 16.2% |

| hours_band | Rows | Actually above 50K | Predicted above 50K | High incomes missed | False alarms | All errors |
|---|---|---|---|---|---|---|
| under 35 | 1,737 | 8.3% | 4.5% | 51.4% | 0.6% | 4.8% |
| 35-40 | 5,235 | 20.6% | 17.3% | 35.9% | 5.1% | 11.5% |
| 41-50 | 1,749 | 40.1% | 38.6% | 24.8% | 14.0% | 18.4% |
| over 50 | 1,037 | 39.7% | 36.6% | 28.6% | 13.8% | 19.7% |
