# How it was built

Decisions in build order. Decider for every entry: Claude, delegated by Michael (Michael set the goal and constraints; Claude made the choices below). Evidence means a file or a run in this repository.

## 1. Single pipeline, all steps fitted inside folds
Column filter, imputer, scaler, SelectKBest and model are one sklearn Pipeline, so no preprocessing sees test data. Rejected: preprocessing the whole dataset first, which leaks. Consequence: a custom column-filter transformer, since sklearn has no one that drops by missing share and near-constant rate together.
**Status:** verified. `analysis.py` (`ColumnFilter`, `build`); filter fitted only on training parts.

## 2. Two evaluations, time-ordered one treated as the honest one
Repeated stratified 5x5 CV as primary, time-ordered 70/30 holdout as secondary. Reason: wafer data drifts, so random CV mixes periods.
**Status:** verified. Both are in `results/metrics.json`; they disagree strongly (CV ROC-AUC 0.736 vs 0.593 for the random forest).

## 3. Thresholds from training data, with the oracle number shown separately
Headline recall at a fixed false-alarm rate uses a threshold from out-of-fold training scores, and the realised false-alarm rate is reported. The test-ROC-curve recall is also kept but labelled optimistic. Rejected: reporting only the ROC-curve number, which flatters the result.
**Status:** verified. Fields `*_trainthr` and `*_roc` in `results/metrics.json`.

## 4. Models and grids kept small
L1 and L2 logistic regression and a random forest, `class_weight` for imbalance, tiny grids. SMOTE not tried (not needed to answer the question, adds a moving part). Consequence: results may move with a larger search; stated in the README.
**Status:** verified. `MODELS` and `K_GRID` in `analysis.py`.

## 5. Headline model picked by CV ROC-AUC
The model shown as headline is the one with the best mean CV ROC-AUC, a different metric from the headline recall, and all three models are reported. Rejected: picking by the headline recall itself, which would select on the quantity being reported.
**Status:** verified. `headline_model_rule` in `results/metrics.json`.

## 6. Two clean top-10 tests instead of one
A top-10 chosen from full-data CV and re-tested on the same data would be optimistic. So: (a) k=10 fixed inside every fold; (b) ten sensors ranked using only the training period, then tested on the later period. Consequence: the top-10 claim is limited to CV, and fails in the time-ordered test.
**Status:** verified. `cv_top10_infold`, `time_holdout_top10_infold`, `time_holdout_top10_fixed_from_train_period` in `results/metrics.json`.

## 7. The negative time-holdout result is reported, not hidden
The time-ordered holdout shows near-chance performance. The README leads with both findings. Rejected: leading only with the CV number.
**Status:** verified. README "Headline" and `results/metrics.json` `time_holdout`.

## 8. Reproducibility
Pinned versions in `requirements.txt`, fixed seed 42, runtime download into a gitignored `data/`. A fresh venv produced identical metrics on two runs (147 to 150 seconds on 16 cores).
**Status:** verified. Two full runs from a clean `.venv` matched on every CV and holdout ROC-AUC printed.

## 9. A scikit-learn deprecation fixed
The first run emitted `penalty` deprecation warnings under scikit-learn 1.9; switched to `l1_ratio`. Metrics were unchanged.
**Status:** verified. Run before and after gave the same ROC-AUCs.

## Open questions
- Would several rolling time splits give a more stable estimate than one 70/30 split with 26 test fails?
- Is the near-chance holdout caused by drift, by too few fails in the training period, or both? Not separated here.
- Are the adjacent top sensors (S431 to S437) one physical signal? Needs sensor names, which the dataset lacks.
- Would a larger hyperparameter search or other models change the picture? Not tested.
