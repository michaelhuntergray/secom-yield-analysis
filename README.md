# secom-yield-analysis

Build log: [HOW-IT-WAS-BUILT.md](HOW-IT-WAS-BUILT.md) | One-page summary: [WRITEUP.md](WRITEUP.md)

A leakage-safe, reproducible attempt to flag failing wafers from process sensor data, using the public UCI SECOM semiconductor dataset (1,567 wafers, 590 anonymised sensors, 104 fails, base fail rate 6.6%). It reports what works and what does not.

## Headline

**Repeated stratified 5-fold CV (5x5) with nested model selection: 29.0% of failing wafers caught at a 10.7% realised false-alarm rate** (mean over 25 folds, std 16.3 points; a random flagger would catch about the same share as its false-alarm rate, so roughly 10.7%). At a 5% target it catches 16.9% (std 13.2) at a 5.2% realised false-alarm rate. In every outer fold an inner CV on that fold's training part picked among logistic L1, logistic L2 and the random forest, and the outer fold scored only the pick; the random forest was picked in 20 of 25 folds, L1 in 3 and L2 in 2.

**That edge does not survive a time-ordered holdout.** Training on the earliest 70% of wafers and testing on the latest 30% (26 fails), the models are close to chance (ROC-AUC 0.536 to 0.584, with wide intervals that include 0.5), and thresholds set on a later slice of the training period caught 3.8% to 26.9% of fails while flagging 7.7% to 20.0% of good wafers, which is about what chance gives. The holdout headline model is the random forest (ROC-AUC 0.548, 95% interval 0.444 to 0.645), chosen using training-period wafers only (see below). Wafer processes drift, so the time-ordered result is the one to believe. See the tables below.

## Charts

![Recall vs false-alarm rate](results/roc_recall_vs_false_alarm.png)

![Top 15 sensors](results/top15_sensors.png)

![Fail rate over time](results/fail_rate_over_time.png)

## Results

All numbers below are copied from `results/metrics.json`. Fail = positive class. Base rate of fails: 6.6% (whole data), 5.5% (time-holdout test period).

### Repeated stratified 5-fold CV (5x5, seed 42)

Mean (std) across 25 folds. The headline row is the nested model-selection procedure described above. The three per-model rows are descriptive: every model scored on the same outer folds, not used for selection. "Recall @ 10% FPR" uses a threshold chosen on the training part only (out-of-sample training scores, with hyperparameter tuning redone inside each calibration fold), applied unchanged to the test fold; the realised false-alarm rate on the test fold is shown next to it. "ROC-curve recall" is read off each test fold's ROC curve at FPR <= 10%; its threshold is chosen with the test labels, so it is not directly comparable with the train-threshold figure, whose realised FPR varies from fold to fold.

| Model | ROC-AUC | PR-AUC (base rate 0.066) | Recall @ 10% FPR (train threshold) | Realised FPR | ROC-curve recall @ 10% | Recall @ 5% FPR (train threshold) | Realised FPR |
|---|---|---|---|---|---|---|---|
| **Nested selection (headline)** | 0.730 (0.041) | 0.172 (0.047) | 29.0% (16.3) | 10.7% (5.6) | 29.3% (9.3) | 16.9% (13.2) | 5.2% (3.3) |
| Logistic L1 | 0.694 (0.047) | 0.157 (0.033) | 26.8% (12.9) | 10.6% (3.7) | 25.8% (7.5) | 15.0% (10.5) | 4.8% (3.1) |
| Logistic L2 | 0.699 (0.046) | 0.158 (0.035) | 30.8% (11.3) | 12.5% (4.3) | 27.9% (8.6) | 17.8% (10.2) | 6.5% (3.1) |
| Random forest | 0.736 (0.040) | 0.183 (0.048) | 30.8% (15.7) | 10.6% (5.5) | 31.0% (9.6) | 18.3% (12.5) | 5.1% (3.0) |

Model selection never sees an outer test fold, so the headline row is an honest estimate of "pick the best of three by inner CV, then deploy". The inner criterion is the best mean ROC-AUC of each model's 3-fold grid search on the training part; the same grid search also tunes that model's hyperparameters. The std is the spread across fold results. Folds from different repeats overlap, so it is not an independent standard error.

### Time-ordered holdout (train earliest 1,097 wafers with 78 fails, test latest 470 wafers with 26 fails)

One split, so no std. ROC-AUC has a 95% bootstrap interval over the test wafers. Inside the training period there is a time-ordered calibration split: each model is tuned and fitted on the earliest 80% of the training wafers (878 wafers, 72 fails), its 10% and 5% false-alarm thresholds are set on the passing wafers of the latest 20% (219 wafers, 6 fails) using that same fitted model, and model and thresholds are applied unchanged to the test period. The test-period scores, ROC-AUC and bootstrap interval all come from that one model.

| Model | ROC-AUC (95% CI) | PR-AUC (base rate 0.055) | Recall @ 10% FPR (train threshold) | Realised FPR | ROC-curve recall @ 10% | ROC-curve recall @ 5% |
|---|---|---|---|---|---|---|
| Logistic L1 | 0.536 (0.443 to 0.632) | 0.058 | 3.8% | 7.7% | 7.7% | 0.0% |
| Logistic L2 | 0.584 (0.481 to 0.689) | 0.071 | 11.5% | 8.3% | 11.5% | 7.7% |
| Random forest | 0.548 (0.444 to 0.645) | 0.062 | 26.9% | 20.0% | 11.5% | 0.0% |

Holdout headline model: the random forest, selected as the model with the highest mean ROC-AUC (0.748, against 0.702 for L1 and 0.674 for L2) in a 5x5 CV run on the earliest 1,097 wafers only, so no test-period label influenced the choice. This is separate from the CV headline above, which selects the model inside each fold.

With 26 test fails, one fail is 3.8 percentage points. All three models sit near the random line: the random forest's 26.9% recall comes with a 20.0% realised false-alarm rate, and the ROC-curve recall columns (threshold chosen with test labels) are 7.7% to 11.5% at 10%. The realised false-alarm rates (7.7%, 8.3% and 20.0% against a 10% target) are closer to the target than in the earlier pooled-threshold version, but the calibration slice holds only 6 fails and about 213 passing wafers, so the thresholds are noisy (the 5% threshold rests on roughly ten wafers). I do not read the mismatch as evidence of drift; the fail-rate chart is the only drift evidence here, and it is visual.

### Top-10 sensors

"Only 10 sensors" is tested with the selection step set to k=10 inside the pipeline, so the 10 sensors are re-chosen from the training part of every fold (and of every inner tuning and threshold-calibration fold) and no validation label shapes the choice. An earlier version fixed a sensor list from the whole training period and reused it in inner folds; a review showed that leaks validation labels into selection, so it was removed.

| Model | CV ROC-AUC | CV recall @ 10% FPR (train threshold) | CV realised FPR | Time-holdout ROC-AUC |
|---|---|---|---|---|
| Logistic L1 | 0.702 | 22.8% | 9.5% | 0.547 |
| Logistic L2 | 0.708 | 26.2% | 10.1% | 0.526 |
| Random forest | 0.706 | 27.2% | 9.9% | 0.492 |

These per-model rows are descriptive and are not used for selection; the time-holdout column uses the same 80/20 calibration split as above.

The sensors picked by the model fitted on the earliest 80% of the training period (same for all three models): S022, S060, S065, S066, S431, S432, S435, S436, S437 and S438. Across 5x3 CV folds of the training period, S060 was picked in 100% of folds and S432 in 80%; per-model frequencies are in `results/metrics.json` (`time_holdout_top10_sensors`). These lists are descriptive and are not scored.

Reading: in cross-validation, 10 sensors do about as well as the tuned 20-to-60-sensor models (for the random forest, 27.2% of fails at a 9.9% realised false-alarm rate, std 10.5 points). In the time-ordered test, 10 sensors do not catch fails reliably either (train-threshold recall 3.8% to 11.5% across the three models at realised false-alarm rates of 7.2% to 12.2%, ROC-AUC 0.492 to 0.547). I do not claim a working 10-sensor detector.

### Sensor ranking

The ranking in the chart is descriptive and is for the random forest (the model the nested selection picked most often), by how often each sensor was selected across the 25 CV folds and by its mean normalised importance (0 when not selected). S060, S104, S130, S511, S029 and S349 were selected in all 25 folds; S432 in 96% and S022, S431 and S435 in 92%. The full top-15 and the top-10 per model are in `results/metrics.json`. Sensor names are 1-based column numbers; the data carries no sensor names.

## Method

One sklearn `Pipeline`, fitted inside every fold, never on the full data:

1. Column filter: drop columns with more than 50% missing values and constant or near-constant columns (most common value covers more than 99% of observed values).
2. Median imputation.
3. Standard scaling.
4. `SelectKBest` (ANOVA F), with k tuned over {20, 60} (or fixed to 10 for the top-10 runs).
5. Model with `class_weight` for the imbalance: logistic regression with L1 penalty (C in 0.01, 0.1, 1), logistic regression with L2 penalty (C in 0.001, 0.01, 0.1), or a random forest (300 trees, `min_samples_leaf` in 1, 5). SMOTE was not tried.

Evaluation:

- **Primary:** `RepeatedStratifiedKFold`, 5 folds x 5 repeats, seed 42. Inside each outer training part, a 3-fold grid search (ROC-AUC) picks hyperparameters for each of the three models, and the model with the best inner ROC-AUC is the one the outer fold scores (nested model selection; pick counts are in `results/metrics.json`, `cv_nested_model_selection`).
- **Thresholds:** for the "train threshold" numbers, the score threshold is the one that gives the target false-alarm rate on out-of-sample scores of the outer training part. Those scores come from a 5-fold `cross_val_predict` over the whole tuning estimator (`GridSearchCV` around the pipeline), so hyperparameters are re-tuned inside every calibration fold and never saw the block they score. The final model is the grid search refitted on all of the outer training part. The threshold is then applied to the test part, and the realised false-alarm rate is reported. The "ROC-curve recall" columns instead read the test fold's own ROC curve at FPR <= target.
- **Secondary:** wafers sorted by timestamp; the first 70% are the training period and the last 30% the test period. Within the training period the earliest 80% fit the model (same tuning) and the latest 20% set the thresholds with that same model; nothing is refitted before the test period is scored.
- Everything is written to `results/metrics.json` with the seed and package versions.

## Reproduce

```
uv venv && uv pip install -r requirements.txt && .venv/bin/python analysis.py
```

Downloads SECOM from UCI at runtime into `data/` (gitignored; raw data is not committed). On a 16-core machine a full run took 477.7 seconds. Tested with Python 3.12.

## Limits

- Only 104 fails (26 in the time-holdout test, 6 in the threshold-calibration slice), so every estimate is noisy. CV std is 6 to 16 points on recall.
- Sensors are anonymised, so nothing here is a physical interpretation. Adjacent columns (S431, S432, S435, S436, S437) rank together, which probably reflects correlated measurements, not 5 independent causes.
- The data comes from one process over about three months and may drift (the fail rate swings visibly between periods; see the fail-rate chart, and note the train and test periods have different base rates). CV mixes time periods, so it is optimistic. The time-ordered holdout is the honest figure.
- The ROC-curve recall columns choose their threshold with the test labels, so they are not directly comparable with the train-threshold figures, whose realised false-alarm rate varies.
- Hyperparameter grids are small on purpose. A bigger search might move the numbers; I did not test that.
- This is an analysis of a public benchmark. It has not been used in production.

## Dataset and license

McCann, M. and Johnston, A. (2008). SECOM. UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/179/secom. Licensed CC BY 4.0. Labels: -1 is pass and 1 is fail in the source; here fail is 1 and pass is 0.

The code in this repository is MIT licensed (see `LICENSE`).
