# How it was built

Decisions in build order. Decider for every entry: Claude, delegated by Michael (Michael set the goal and constraints; Claude made the choices below). Evidence means a file or a run in this repository.

## 1. Single pipeline, all steps fitted inside folds
Column filter, imputer, scaler, SelectKBest and model are one sklearn Pipeline, so no preprocessing sees test data. Rejected: preprocessing the whole dataset first, which leaks. Consequence: a custom column-filter transformer, since sklearn has no one that drops by missing share and near-constant rate together.
**Status:** verified. `analysis.py` (`ColumnFilter`, `build`); filter fitted only on training parts.

## 2. Two evaluations, time-ordered one treated as the honest one
Repeated stratified 5x5 CV as primary, time-ordered 70/30 holdout as secondary. Reason: wafer data drifts, so random CV mixes periods.
**Status:** verified. Both are in `results/metrics.json`; they disagree strongly (CV ROC-AUC 0.730 for the nested selection procedure vs 0.548 for the holdout random forest).

## 3. Thresholds from training data, with the oracle number shown separately
Headline recall at a fixed false-alarm rate uses a threshold from out-of-sample training scores, and the realised false-alarm rate is reported. The test-ROC-curve recall is also kept, described as threshold-chosen-with-test-labels and not directly comparable (an earlier draft called it an upper bound; the metrics contradict that, since L2 CV scores 27.9% by ROC curve and 30.8% by train threshold). Rejected: reporting only the ROC-curve number.
**Status:** verified. Fields `*_trainthr` and `*_roc` in `results/metrics.json`.

## 4. Models and grids kept small
L1 and L2 logistic regression and a random forest, `class_weight` for imbalance, tiny grids. SMOTE not tried (not needed to answer the question, adds a moving part). Consequence: results may move with a larger search; stated in the README.
**Status:** verified. `MODELS` and `K_GRID` in `analysis.py`.

## 5. Headline model chosen without the labels it is scored on
First version: the CV headline was the model with the best mean ROC-AUC over the 5x5 CV on all wafers, reported from that same CV. Review (Codex) found that choice used labels from the 470 holdout wafers, so the holdout headline was moved to a 5x5 CV on the earliest 1,097 wafers only (still the random forest). A later review found the CV headline still used its own evaluation labels. Reversed: the CV headline is now nested model selection. In each outer fold the model with the best inner 3-fold ROC-AUC on that fold's training part is picked, and the outer fold scores only the pick (random forest in 20 of 25 folds, L1 in 3, L2 in 2). The per-model CV table is kept as descriptive and is not used for selection. Rejected: shrinking to 3x5 repeats (not needed, runtime stayed about 8 minutes, because the inner scores come from the grid searches already being run). Decider: Claude, on reviewer findings.
**Status:** verified. `cv_nested_model_selection` (summary, `pick_counts`, `rule`), `time_holdout_headline_model` and its rule in `results/metrics.json`.

## 6. Top-10 sensors selected inside the folds only
A top-10 chosen from full-data CV and re-tested on the same data would be optimistic, so selection is a pipeline step (k=10) refitted in every fold. A second variant (ten sensors fixed from the training period, then reused in inner tuning and threshold calibration) was built first and removed after review: the fixed list had seen the labels of the inner validation folds. Rejected: keeping it with a caveat. The sensor lists are now reported as descriptive selection frequencies and a final refit. Consequence: the top-10 claim is limited to CV, and fails in the time-ordered test. An audit of the rest of the code for the same pattern (selection fitted on data a later step validates on) found none; one soft spot was found, disclosed, and then fixed after a further review: hyperparameters were tuned on all of a training part and the resulting scores were called out-of-fold when used to set the threshold. Threshold scores now come from `cross_val_predict` over the whole `GridSearchCV(pipeline)`, so tuning is redone inside every calibration fold; the final model is still the grid search refitted on the whole training part. Cost: runtime rose from about 170 s to several minutes. Effect: train-threshold recall estimates fell slightly for some models and their spread widened (std 11 to 16 points). The CV-headline model-choice caveat from that review is also fixed (entry 5).
**Status:** verified. `cv_top10_infold`, `time_holdout_top10_infold`, `time_holdout_top10_sensors` in `results/metrics.json`; `analysis.py` has no fixed-list code path and calibrates thresholds with `cross_val_predict(clone(gs), ...)`.

## 6b. Holdout thresholds come from the model that is scored
The holdout first pooled out-of-fold training scores from several fold models and applied that quantile to a model refitted on all training wafers; the score scales of the two can differ (Codex review). Replaced by a time-ordered calibration split inside the training period: tune and fit on the earliest 80% of training wafers, set the 10% and 5% false-alarm thresholds on the latest 20% with that same model, apply both unchanged to the test period. The bootstrap interval now uses the scores of that same model. Consequence: only 6 fails and about 213 passing wafers calibrate the thresholds, so they are noisy; realised false-alarm rates became 7.7%, 8.3% and 20.0% and holdout ROC-AUC moved to 0.536, 0.584 and 0.548 (fits on 80% of the training period instead of 100%). The earlier README sentence that the false-alarm mismatch "suggests" drift was removed; the new numbers do not support it. Rejected: keeping the pooled threshold with a caveat.
**Status:** verified. `time_holdout` in `analysis.py`; `time_holdout` entries (`n_fit`, `n_calibration`) in `results/metrics.json`.

## 7. The negative time-holdout result is reported, not hidden
The time-ordered holdout shows near-chance performance. The README leads with both findings. Rejected: leading only with the CV number.
**Status:** verified. README "Headline" and `results/metrics.json` `time_holdout`.

## 8. Reproducibility
Pinned versions in `requirements.txt`, fixed seed 42, runtime download into a gitignored `data/`. The first version was not fully deterministic: liblinear (L1 logistic) had no random_state and its holdout ROC-AUC moved in the third decimal between runs. Fixed by seeding it. Two full runs afterwards produced identical `results/metrics.json` apart from runtime (about 480 seconds on 16 cores, with nested threshold calibration and nested model selection).
**Status:** verified. Dictionary comparison of two runs' metrics, runtime excluded.

## 9. A scikit-learn deprecation fixed
The first run emitted `penalty` deprecation warnings under scikit-learn 1.9; switched to `l1_ratio`. Metrics were unchanged.
**Status:** verified. Run before and after gave the same ROC-AUCs.

## Open questions
- Would several rolling time splits give a more stable estimate than one 70/30 split with 26 test fails?
- Is the near-chance holdout caused by drift, by too few fails in the training period, or both? Not separated here.
- Are the adjacent top sensors (S431 to S437) one physical signal? Needs sensor names, which the dataset lacks.
- Would a larger hyperparameter search or other models change the picture? Not tested.
