# Design decisions

One page on the choices made in this project and their trade-offs.

| Date | Decision | Options considered | Why this one |
|---|---|---|---|
| 2026-11-15 | Dataset: Adult Income (classification) | Adult Income, California housing (regression) | Mixed numeric and categorical columns, missing values and unbalanced classes give more real cleaning and feature work to show. |
| 2026-11-15 | Report F1, ROC AUC, RMSE and MAE, each with its meaning and its weakness | one main metric | Seeing them side by side shows what each one hides. RMSE and MAE are regression metrics, so here they are computed on the predicted probability against the true 0 or 1. |
| 2026-11-15 | The model is chosen by cross-validated F1 | F1, ROC AUC, RMSE, MAE | One rule is needed to pick a model. The classes are unbalanced and the output is a yes/no answer. It is one setting (`selection_metric`), and in this run every one of the four metrics picks the same model. |
| 2026-11-15 | Training runs from a Python script (`python scripts/train.py`) | a script, a Makefile | It works on Windows, macOS and Linux without installing `make`. |
| 2026-11-15 | Train on a laptop CPU, kept light: one model at a time, at most 2 threads, near-default settings, no tuning | a parameter search, or training on a stronger machine | A working, reproducible pipeline comes first. The whole run takes under a minute. Heavier settings can be set in `config.py` later. |
| 2026-11-15 | Three models plus a baseline: logistic regression, random forest, gradient boosting | fewer models, or others (SVM, k-nearest neighbours) | They go from simple and linear to flexible, and all train in seconds. The baseline shows what "no skill" scores on each metric. |
| 2026-11-15 | Features are a `ColumnTransformer` inside each model's `Pipeline` | preparing the table once, before splitting | Medians, scales and category lists are learned from training rows only, so nothing leaks into the validation folds or the test rows. |
| 2026-11-15 | Missing categories become their own category, "missing" | fill with the most common value, or drop the rows | In the notebook, people with a missing work class or occupation earn above 50K far less often (about 9%) than average (24%). The gap carries information. |
| 2026-11-15 | Categories under 1% of the rows are grouped | one column per category | `native_country` has 39 rare values. Grouping them avoids dozens of nearly empty columns. |
| 2026-11-15 | Capital gain and loss get `log(1 + x)` before scaling | scaling only | They are zero for over 90% of people and have a very long tail. |
| 2026-11-15 | `fnlwgt` and `education` are dropped | keeping them | `fnlwgt` is a census sampling weight, not a property of the person. `education` repeats `education_num`. |
| 2026-11-15 | Both source files are joined, then split 80/20 with a fixed seed | keeping the original train/test files | One stratified split is simple to explain and keeps the class balance equal in both parts. The cost: results are not directly comparable with papers that use the original split. |
| 2026-11-15 | 52 exact duplicate rows are removed | keeping them | The same row in both the training and the test part would flatter the test score. Some may be different people with equal answers; that cannot be known. |
| 2026-11-15 | The raw data and the model file are not stored in the repository | committing them | The data has its own license and source, and both are rebuilt by one command. The download checks a checksum for each file. |
| 2026-11-15 | `reports/` (metrics and figures) is stored in the repository | generated only | A visitor sees the results without running anything, and a rerun shows at once whether a number changed. |

## Trade-offs

- **Light settings rather than the best score.** The run is fast and gentle on the CPU. The scores are below what tuning would reach.
- **A fixed cut-off of 0.5.** Simple to explain. It leaves recall low (0.68); choosing the cut-off on validation data is a next step.
- **Sensitive columns are kept as inputs.** That matches common practice with this dataset and makes the group differences visible in the error analysis. A model for real use would need a fairness review first.
- **Fixed, pinned versions.** The same numbers come out on every run. Other library versions may change the last decimals.
