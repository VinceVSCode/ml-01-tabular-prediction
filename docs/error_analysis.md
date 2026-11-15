# Error analysis

Where the chosen model (gradient boosting, cut-off 0.5) is wrong on the 9,758 test rows. Every number is from [reports/metrics.md](../reports/metrics.md), which `python scripts/train.py` rewrites.

## The two kinds of error

| | Predicted 50K or less | Predicted above 50K |
|---|---|---|
| **Actually 50K or less** (7,422) | 6,966 correct | 456 false alarms |
| **Actually above 50K** (2,336) | 753 missed | 1,583 correct |

- The model is wrong for 1,209 people (12.4%).
- It **misses more than it over-predicts**: 32% of high incomes are missed, while only 6% of low incomes are flagged by mistake. With three low incomes for every high one, answering "50K or less" is the safer guess, and at a cut-off of 0.5 the model leans that way.

## Errors by group

"Missed" is the share of real high incomes the model did not find. "False alarms" is the share of real low incomes it flagged.

| Group | Rows | Actually above 50K | Missed | False alarms | All errors |
|---|---|---|---|---|---|
| Women | 3,326 | 11.5% | 43.1% | 1.6% | 6.4% |
| Men | 6,432 | 30.4% | 30.1% | 9.1% | 15.5% |
| Age under 25 | 1,663 | 1.5% | 72.0% | 0.1% | 1.1% |
| Age 25-34 | 2,463 | 15.6% | 40.3% | 3.6% | 9.3% |
| Age 35-44 | 2,446 | 34.3% | 30.8% | 9.8% | 17.0% |
| Age 45-54 | 1,783 | 38.0% | 24.9% | 12.8% | 17.4% |
| Age 55-64 | 958 | 32.6% | 35.9% | 10.7% | 18.9% |
| Age 65 and over | 445 | 21.8% | 42.3% | 3.2% | 11.7% |
| Below high school | 1,292 | 5.8% | 69.3% | 0.1% | 4.1% |
| High school | 3,115 | 16.1% | 62.0% | 2.6% | 12.1% |
| Some college | 2,909 | 18.6% | 39.2% | 6.3% | 12.4% |
| Bachelor | 1,625 | 43.3% | 17.1% | 17.9% | 17.5% |
| Above bachelor | 817 | 63.3% | 11.4% | 24.3% | 16.2% |
| Under 35 hours a week | 1,737 | 8.3% | 51.4% | 0.6% | 4.8% |
| 35-40 hours | 5,235 | 20.6% | 35.9% | 5.1% | 11.5% |
| 41-50 hours | 1,749 | 40.1% | 24.8% | 14.0% | 18.4% |
| Over 50 hours | 1,037 | 39.7% | 28.6% | 13.8% | 19.7% |

## What the table shows

1. **The model follows the base rate of each group.** Where high incomes are rare (women, the young, less schooling, part-time work), it rarely predicts one. So it makes few false alarms there, but misses most of the high earners who do exist: 72% of those under 25 and 69% of those without a high-school diploma. Some of these groups are small, so the exact shares are uncertain (only about 25 people under 25 earn above 50K in the test rows).
2. **A low error rate can hide a weak model.** The group "under 25" has the lowest error rate (1.1%) only because almost nobody in it earns above 50K. This is the weakness of accuracy, seen inside one group.
3. **The opposite happens where high incomes are common.** For people with more than a bachelor's degree, the model finds 89% of the high incomes but wrongly flags 24% of the others.
4. **Women and men are treated differently.** A high-earning woman is missed 43% of the time, a high-earning man 30%. The model has learned the pattern in the 1994 data, where 11% of women and 30% of men earned above 50K. This is a reason not to use it for decisions about people.

## What could reduce these errors (not done in this version)

- Choose the cut-off on validation data instead of using 0.5. A lower cut-off finds more high incomes at the cost of more false alarms.
- Check the calibration of the probabilities, overall and by group.
- Train without `sex`, `race` and `native_country` and compare the errors by group. Other columns (such as `relationship`, with the values "Husband" and "Wife") still carry part of that information.
- Tune the model. More capacity may help most in the groups where the signal is weak.
