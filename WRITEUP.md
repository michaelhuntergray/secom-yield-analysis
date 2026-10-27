# SECOM wafer-fail detection: what the data supports

**Problem.** A fab records hundreds of sensor readings per wafer, and a small share of wafers fail at end of line. The public UCI SECOM dataset has 1,567 wafers, 590 anonymised sensors and 104 fails (a 6.6% base rate), with timestamps. The question: can the sensor data flag likely fails early enough to be useful, and how well can I trust the answer?

**Approach.** One sklearn pipeline (drop empty, constant and near-constant columns; median-impute; scale; select the 20 or 60 best sensors; fit a model), with every step fitted inside each fold so nothing from the test data leaks in. Three models: L1 and L2 logistic regression and a random forest, all weighted for the class imbalance. Two evaluations: repeated stratified 5-fold cross-validation (5x5, fixed seed 42), and a time-ordered holdout that trains on the earliest 70% of wafers and tests on the latest 30%. The main number is recall of fails at a fixed false-alarm rate, meaning the share of failing wafers caught when 10% of good wafers are flagged by mistake. The alarm threshold is set on training data only.

**Result.**
- Cross-validation: the random forest caught 30.7% of fails at a 10.0% false-alarm rate (std 10.2 points across 25 folds), and 18.9% at 5%. A random flagger catches 10% at a 10% false-alarm rate, so this is about three times better than chance, with ROC-AUC 0.736 and PR-AUC 0.183 against a 0.066 base rate.
- Ten sensors chosen inside the folds did about as well in cross-validation (random forest: 27.5% of fails at a 9.4% realised false-alarm rate).
- Time-ordered holdout (26 fails in the test period): ROC-AUC 0.492, 0.619 and 0.593 for L1, L2 and the random forest, with bootstrap intervals that include 0.5 for all three. The random forest is the holdout headline model, chosen by cross-validation on the training period only. Thresholds set on training data caught 0.0%, 3.8% and 0.0% of fails. I could not show a usable detector once time order is respected.

**What it would mean on a fab line.** A flagger that finds a third of excursions at a 1-in-10 false-alarm cost could, if it held up, prioritise metrology or sampling. These results say it would not hold up unless it is retrained and monitored as the process drifts, and that sensor-level conclusions need domain knowledge: the sensors are anonymised, so a ranking such as S060, S104 and S130 (selected in every CV fold) is a place to look, not a root cause. Cross-validation alone would have overstated the result; the time-ordered test is what caught that.

**Limits.** 104 fails is small, and 26 in the test period means one wafer moves recall by 3.8 points. The data is a single process from 2008. The grids are small. No production use is claimed.

**Next steps.** (1) Rolling-origin evaluation with several time splits instead of one. (2) Drift handling: retrain on a moving window and track score shift. (3) Cost-based thresholds using real scrap and inspection costs. (4) With sensor names or process-step labels, group correlated sensors and test whether the adjacent-column cluster (S431 to S437) is one physical signal. (5) Try the same pipeline on a larger dataset with more fails.

Code, charts and numbers: see `README.md` and `results/metrics.json`. Dataset: McCann and Johnston (2008), UCI Machine Learning Repository, CC BY 4.0.
