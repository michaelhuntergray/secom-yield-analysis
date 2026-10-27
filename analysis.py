"""Leakage-safe yield-excursion analysis of the UCI SECOM dataset.

Run:  python analysis.py
Writes results/metrics.json and PNG charts into results/.
Every preprocessing step (column filtering, imputation, scaling, feature
selection) lives inside one sklearn Pipeline and is fitted inside each fold.
"""
import io
import json
import time
import urllib.request
import zipfile
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_SPLITS, N_REPEATS = 5, 5
TRAIN_FRAC = 0.70
CAL_SPLIT = 0.80  # earliest share of the training period used to fit; the rest calibrates thresholds
FPR_TARGETS = (0.10, 0.05)
N_JOBS = 16
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"
URLS_ZIP = ["https://archive.ics.uci.edu/static/public/179/secom.zip"]
URLS_RAW = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
FPR_GRID = np.linspace(0, 1, 201)


# --------------------------------------------------------------------- data
def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "secom-yield-analysis"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def download():
    """Fetch SECOM into data/ (gitignored) unless already cached."""
    DATA_DIR.mkdir(exist_ok=True)
    need = ["secom.data", "secom_labels.data"]
    if all((DATA_DIR / n).exists() for n in need):
        return
    try:
        for url in URLS_ZIP:
            blob = _fetch(url)
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for n in need + ["secom.names"]:
                    (DATA_DIR / n).write_bytes(z.read(n))
        return
    except Exception as e:  # fall back to the raw files
        print(f"zip download failed ({e}); trying raw files")
    for n in need + ["secom.names"]:
        (DATA_DIR / n).write_bytes(_fetch(URLS_RAW + n))


def load():
    download()
    X = pd.read_csv(DATA_DIR / "secom.data", sep=r"\s+", header=None, na_values="NaN")
    lab = pd.read_csv(DATA_DIR / "secom_labels.data", sep=" ", header=None,
                      quotechar='"', names=["label", "ts"])
    ts = pd.to_datetime(lab["ts"], format="%d/%m/%Y %H:%M:%S")
    y = (lab["label"].to_numpy() == 1).astype(int)  # 1 = fail, 0 = pass
    order = np.argsort(ts.to_numpy(), kind="stable")  # chronological
    return X.to_numpy(float)[order], y[order], ts.to_numpy()[order]


# ---------------------------------------------------------------- pipeline
class ColumnFilter(BaseEstimator, TransformerMixin):
    """Drop columns with > max_missing missing share, or constant / near-constant
    (most common value covers > max_top_freq of observed values). Fitted on
    training data only."""

    def __init__(self, max_missing=0.5, max_top_freq=0.99):
        self.max_missing = max_missing
        self.max_top_freq = max_top_freq

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        keep = np.zeros(X.shape[1], bool)
        for j in range(X.shape[1]):
            col = X[:, j]
            obs = col[~np.isnan(col)]
            if len(obs) == 0 or (1 - len(obs) / len(col)) > self.max_missing:
                continue
            _, counts = np.unique(obs, return_counts=True)
            if len(counts) < 2 or counts.max() / len(obs) > self.max_top_freq:
                continue
            keep[j] = True
        self.support_ = keep
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.support_]


MODELS = {
    "logreg_l1": (
        lambda: LogisticRegression(l1_ratio=1, solver="liblinear",
                                   class_weight="balanced", max_iter=2000,
                                   random_state=SEED),
        {"model__C": [0.01, 0.1, 1.0]},
    ),
    "logreg_l2": (
        lambda: LogisticRegression(l1_ratio=0, solver="lbfgs",
                                   class_weight="balanced", max_iter=2000),
        {"model__C": [0.001, 0.01, 0.1]},
    ),
    "random_forest": (
        lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                       n_jobs=1, random_state=SEED),
        {"model__min_samples_leaf": [1, 5]},
    ),
}
K_GRID = [20, 60]


def build(name, k=None):
    """Return (pipeline, param_grid). k=None tunes k in K_GRID; int fixes it."""
    make, grid = MODELS[name]
    pipe = Pipeline([
        ("filter", ColumnFilter()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k if k else K_GRID[0])),
        ("model", make()),
    ])
    grid = dict(grid)
    if k is None:
        grid["select__k"] = K_GRID
    return pipe, grid


def score(est, X):
    m = est.named_steps["model"]
    return est.decision_function(X) if isinstance(m, LogisticRegression) \
        else est.predict_proba(X)[:, 1]


def method_for(pipe):
    return "decision_function" if isinstance(pipe.named_steps["model"], LogisticRegression) \
        else "predict_proba"


# ----------------------------------------------------------------- metrics
def recall_at_fpr_oracle(y, s, fpr_t):
    """Max recall with test-fold FPR <= fpr_t, read off the test ROC curve."""
    # drop_intermediate=False: the default can drop the best point under the target
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    return float(tpr[fpr <= fpr_t].max())


def _self_check():
    """Tie case where the default roc_curve point dropping loses the best point:
    1/3 recall is achievable at FPR 0.5."""
    r = recall_at_fpr_oracle(np.array([1, 0, 1, 0, 1, 0]), np.array([3, 3, 2, 2, 1, 1]), 0.5)
    assert abs(r - 1 / 3) < 1e-12, r


def fit_and_eval(pipe, grid, Xtr, ytr, Xte, yte, seed, orig_cols=None):
    """Tune on the training part only, set thresholds on training-part
    out-of-sample scores only (tuning nested inside), evaluate on the held-out part."""
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1, refit=True)
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    s_te = score(best, Xte)
    # thresholds: out-of-sample scores on the training part. The whole tuning
    # estimator (GridSearchCV over the pipeline) is cloned and refitted inside
    # every calibration fold, so hyperparameters never saw the fold they score.
    oof = cross_val_predict(clone(gs), Xtr, ytr, method=method_for(best),
                            cv=StratifiedKFold(5, shuffle=True, random_state=seed + 1))
    if oof.ndim == 2:
        oof = oof[:, 1]
    res = _score_part(gs, best, yte, s_te, oof[ytr == 0])
    # inner-CV ROC-AUC of the tuned configuration, used only for nested model selection
    res["inner_cv_roc_auc"] = float(gs.best_score_)
    return res


def _score_part(gs, best, yte, s_te, neg):
    """Metrics of one fitted model on a held-out part, with thresholds taken
    from `neg` (scores of passing wafers that the model did not train on)."""
    res = {
        "roc_auc": float(roc_auc_score(yte, s_te)),
        "pr_auc": float(average_precision_score(yte, s_te)),
        "n_test": int(len(yte)), "n_test_fail": int(yte.sum()),
        "best_params": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in gs.best_params_.items()},
    }
    for t in FPR_TARGETS:
        tag = f"{int(t * 100)}"
        res[f"recall_at_fpr{tag}_roc"] = recall_at_fpr_oracle(yte, s_te, t)
        thr = np.quantile(neg, 1 - t)
        alarm = s_te > thr
        res[f"recall_at_fpr{tag}_trainthr"] = float(alarm[yte == 1].mean())
        res[f"realised_fpr{tag}_trainthr"] = float(alarm[yte == 0].mean())
    fpr, tpr, _ = roc_curve(yte, s_te, drop_intermediate=False)
    res["_tpr_grid"] = np.interp(FPR_GRID, fpr, tpr).tolist()
    # which original columns were used and how important were they
    kept = np.flatnonzero(best.named_steps["filter"].support_)
    sel = best.named_steps["select"].get_support(indices=True)
    orig = kept[sel]
    m = best.named_steps["model"]
    imp = np.abs(m.coef_.ravel()) if isinstance(m, LogisticRegression) else m.feature_importances_
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    res["_orig"], res["_imp"] = orig.tolist(), imp.tolist()
    res["_s_te"] = s_te
    return res


def _cv_fold(name, k, X, y, tr, te, seed):
    pipe, grid = build(name, k)
    return fit_and_eval(pipe, grid, X[tr], y[tr], X[te], y[te], seed)


def run_cv(name, X, y, k=None, n_repeats=N_REPEATS):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=SEED)
    return Parallel(n_jobs=N_JOBS)(
        delayed(_cv_fold)(name, k, X, y, tr, te, SEED + i)
        for i, (tr, te) in enumerate(cv.split(X, y)))


def summarise(folds):
    keys = ["roc_auc", "pr_auc"] + [f"{p}{int(t * 100)}{s}" for t in FPR_TARGETS for p, s in [
        ("recall_at_fpr", "_roc"), ("recall_at_fpr", "_trainthr"), ("realised_fpr", "_trainthr")]]
    out = {"n_folds": len(folds)}
    for k in keys:
        v = np.array([f[k] for f in folds])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1))}
    return out


def sensor_ranking(folds, n_cols, top=15):
    cnt, imp = np.zeros(n_cols), np.zeros(n_cols)
    for f in folds:
        cnt[f["_orig"]] += 1
        imp[f["_orig"]] += np.array(f["_imp"])
    freq, mimp = cnt / len(folds), imp / len(folds)
    order = np.lexsort((-mimp, -freq))[:top]
    return [{"sensor": f"S{j + 1:03d}", "col_index_1based": int(j + 1),
             "selection_frequency": float(freq[j]), "mean_importance": float(mimp[j])}
            for j in order]


def mean_curve(folds):
    a = np.array([f["_tpr_grid"] for f in folds])
    return a.mean(0), a.std(0)


# ------------------------------------------------------------- time holdout
def time_holdout(name, X, y, k, seed=SEED):
    """Time-ordered holdout with a time-ordered calibration split inside the
    training period: tune and fit on the earliest CAL_SPLIT of the training
    wafers, set the FPR thresholds on the remaining (latest) training wafers
    with that same fitted model, then apply model and thresholds unchanged to
    the test period."""
    n_tr = int(round(TRAIN_FRAC * len(y)))
    n_fit = int(round(CAL_SPLIT * n_tr))
    pipe, grid = build(name, k)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc",
                      cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                      n_jobs=1, refit=True)
    gs.fit(X[:n_fit], y[:n_fit])
    best = gs.best_estimator_
    s_cal = score(best, X[n_fit:n_tr])
    res = _score_part(gs, best, y[n_tr:], score(best, X[n_tr:]),
                      s_cal[y[n_fit:n_tr] == 0])
    res["n_fit"], res["n_fit_fail"] = n_fit, int(y[:n_fit].sum())
    res["n_calibration"], res["n_calibration_fail"] = n_tr - n_fit, int(y[n_fit:n_tr].sum())
    res["n_train_period"], res["n_train_period_fail"] = n_tr, int(y[:n_tr].sum())
    return res


def bootstrap_auc_ci(res, y, n_boot=1000):
    """95% bootstrap CI of the holdout ROC-AUC (resampling test wafers only,
    scores of the very model that was evaluated)."""
    n_tr = int(round(TRAIN_FRAC * len(y)))
    s, yt = res["_s_te"], y[n_tr:]
    rng = np.random.default_rng(SEED)
    aucs = []
    while len(aucs) < n_boot:
        i = rng.integers(0, len(yt), len(yt))
        if 0 < yt[i].sum() < len(i):
            aucs.append(roc_auc_score(yt[i], s[i]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def top10_selection_report(name, X, y):
    """Descriptive only, nothing here is scored: how often each sensor is picked
    by the in-pipeline k=10 selector across 5x3 CV folds of the training period
    (selection is refitted inside every fold)."""
    n_tr = int(round(TRAIN_FRAC * len(y)))
    folds = run_cv(name, X[:n_tr], y[:n_tr], k=10, n_repeats=3)
    return sensor_ranking(folds, X.shape[1], top=10)


# ------------------------------------------------------------------ charts
COL = {"logreg_l1": "#1f77b4", "logreg_l2": "#2ca02c", "random_forest": "#d62728", "nested": "#000000"}
LAB = {"logreg_l1": "Logistic L1", "logreg_l2": "Logistic L2", "random_forest": "Random forest",
       "nested": "Nested model selection (headline)"}


def chart_roc(cv_curves, th_curves, headline, headline_th, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, title, curves, hl in [
        (axes[0], f"Repeated stratified {N_SPLITS}-fold CV ({N_SPLITS}x{N_REPEATS}), mean of folds", cv_curves, headline),
        (axes[1], "Time-ordered holdout (train earliest 70%, test latest 30%)", th_curves, headline_th),
    ]:
        for name, (m, sd) in curves.items():
            lw = 2.6 if name == hl else 1.4
            ax.plot(FPR_GRID, m, color=COL[name], lw=lw, label=LAB[name])
            if sd is not None:
                ax.fill_between(FPR_GRID, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1),
                                color=COL[name], alpha=0.08)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
        ax.axvline(0.10, color="gray", ls=":", lw=1.2)
        ax.text(0.105, 0.03, "10% false alarms", color="gray", fontsize=8)
        ax.set_xlim(0, 0.4)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False-alarm rate (passing wafers flagged)")
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Recall of failing wafers")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("SECOM: recall of fails vs false-alarm rate", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_sensors(top, headline, path):
    names = [t["sensor"] for t in top][::-1]
    fr = [t["selection_frequency"] for t in top][::-1]
    im = [t["mean_importance"] for t in top][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)
    axes[0].barh(names, fr, color=COL[headline])
    axes[0].set_xlabel(f"Selection frequency across {N_SPLITS * N_REPEATS} CV folds")
    axes[0].set_xlim(0, 1)
    axes[1].barh(names, im, color="#7f7f7f")
    axes[1].set_xlabel("Mean normalised importance (0 when not selected)")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("Anonymised sensor (1-based column number)")
    fig.suptitle(f"Top 15 sensors, {LAB[headline]} (most often picked model; descriptive)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_time(y, ts, path, n_tr, bins=16):
    edges = np.linspace(0, len(y), bins + 1).astype(int)
    rate = [y[a:b].mean() for a, b in zip(edges[:-1], edges[1:])]
    mid = [ts[(a + b) // 2] for a, b in zip(edges[:-1], edges[1:])]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mid, np.array(rate) * 100, "o-", color="#d62728")
    ax.axhline(y.mean() * 100, color="gray", ls="--", lw=1, label=f"Overall {y.mean() * 100:.1f}%")
    ax.axvline(ts[n_tr], color="k", ls=":", label="Train/test split (70%)")
    ax.set_xlabel("Date of processing")
    ax.set_ylabel("Fail rate (%), ~98 wafers per point")
    ax.set_title("SECOM fail rate over time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- main
def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    return o


def main():
    t0 = time.time()
    _self_check()
    OUT_DIR.mkdir(exist_ok=True)
    X, y, ts = load()
    n, p = X.shape
    print(f"{n} wafers, {p} sensors, {int(y.sum())} fails ({y.mean():.3%})")
    filt = ColumnFilter().fit(X)
    metrics = {
        "seed": SEED,
        "versions": {pk: version(pk) for pk in
                     ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib"]},
        "dataset": {"n_wafers": n, "n_sensors": p, "n_fail": int(y.sum()), "n_pass": int((1 - y).sum()),
                    "fail_base_rate": float(y.mean()),
                    "n_sensors_kept_by_filter_on_all_data_descriptive_only": int(filt.support_.sum()),
                    "first_timestamp": str(ts[0])[:19], "last_timestamp": str(ts[-1])[:19]},
        "protocol": {
            "cv": f"RepeatedStratifiedKFold {N_SPLITS}x{N_REPEATS}, random_state={SEED}",
            "inner_tuning": "GridSearchCV 3-fold, scoring roc_auc, on the training part only",
            "threshold_trainthr": "CV results: score threshold at the target FPR of out-of-sample scores on the training part, from cross_val_predict (5-fold) over the whole GridSearchCV(pipeline), so tuning is redone inside every calibration fold; applied unchanged to the test part",
            "threshold_time_holdout": f"time-ordered calibration split inside the training period: model tuned and fitted on the earliest {int(CAL_SPLIT * 100)}% of the training wafers, thresholds set on the passing wafers of the latest {int(round((1 - CAL_SPLIT) * 100))}% with that same fitted model; model and thresholds applied unchanged to the test period",
            "cv_headline": "nested model selection: per outer fold, inner 3-fold CV on the training part picks among L1, L2 and random forest; the outer fold scores the pick (see cv_nested_model_selection)",
            "threshold_roc": "best recall with FPR <= target read off each test part's ROC curve; the threshold is chosen with test labels, so it is not directly comparable with the train-threshold figure, whose realised FPR varies",
            "k_grid": K_GRID, "time_train_fraction": TRAIN_FRAC,
            "std": "sample std across the fold results; folds across repeats overlap, so it is not an independent-sample standard error",
        },
        "cv": {}, "cv_top10_infold": {}, "time_holdout": {}, "time_holdout_top10_infold": {},
    }

    cv_folds, curves_cv = {}, {}
    for name in MODELS:
        t = time.time()
        folds = run_cv(name, X, y)
        cv_folds[name] = folds
        metrics["cv"][name] = summarise(folds)
        curves_cv[name] = mean_curve(folds)
        print(f"CV {name}: AUC {metrics['cv'][name]['roc_auc']['mean']:.3f} ({time.time() - t:.0f}s)")
    metrics["cv_per_model_note"] = ("descriptive only: every model scored on the same outer folds; "
                                    "not used for selection (see cv_nested_model_selection)")

    # CV headline = nested model selection. In every outer fold the model with the
    # highest inner-CV ROC-AUC (3-fold, on that fold's training part only) is picked
    # and the outer fold scores only that pick. Folds and seeds are identical across
    # models, so fold i of each model is the same outer split.
    names = list(MODELS)
    nested = []
    for i in range(len(cv_folds[names[0]])):
        cand = {nm: cv_folds[nm][i] for nm in names}
        assert len({c["n_test"] for c in cand.values()}) == 1
        pick = max(names, key=lambda nm: cand[nm]["inner_cv_roc_auc"])
        f = dict(cand[pick])
        f["_picked"] = pick
        nested.append(f)
    picks = {nm: sum(f["_picked"] == nm for f in nested) for nm in names}
    metrics["cv_nested_model_selection"] = {
        **summarise(nested),
        "pick_counts": picks,
        "rule": ("in each outer fold the model with the highest inner 3-fold CV ROC-AUC on that fold's "
                 "training part is picked (the inner grid search also tunes its hyperparameters); the "
                 "outer fold scores only the picked model, with thresholds calibrated on the training "
                 "part as for every other CV result"),
    }
    curves_cv["nested"] = mean_curve(nested)
    print("nested picks", picks, f"AUC {metrics['cv_nested_model_selection']['roc_auc']['mean']:.3f}")
    # descriptive sensor ranking for the most frequently picked model (ties: higher mean AUC)
    headline = max(names, key=lambda nm: (picks[nm], metrics["cv"][nm]["roc_auc"]["mean"]))
    metrics["sensor_ranking_model"] = headline

    top = sensor_ranking(cv_folds[headline], p)
    metrics["top15_sensors"] = {"model": headline, "ranking": top}
    metrics["top10_sensors_by_model"] = {nm: sensor_ranking(cv_folds[nm], p, top=10) for nm in MODELS}

    # in-fold top-10 (k=10 inside every fold) for every model
    for name in MODELS:
        folds = run_cv(name, X, y, k=10)
        metrics["cv_top10_infold"][name] = summarise(folds)
        print(f"CV top-10 in-fold {name}: AUC {metrics['cv_top10_infold'][name]['roc_auc']['mean']:.3f}")

    # time-ordered holdout
    n_tr = int(round(TRAIN_FRAC * n))
    metrics["time_split"] = {"n_train": n_tr, "n_test": n - n_tr,
                             "train_fails": int(y[:n_tr].sum()), "test_fails": int(y[n_tr:].sum()),
                             "train_fail_rate": float(y[:n_tr].mean()),
                             "test_fail_rate": float(y[n_tr:].mean()),
                             "split_timestamp": str(ts[n_tr])[:19]}
    # holdout headline model: chosen by CV on the training-period wafers only,
    # so no label from the test period influences the choice
    n_tr_sel = int(round(TRAIN_FRAC * n))
    sel_auc = {nm: summarise(run_cv(nm, X[:n_tr_sel], y[:n_tr_sel]))["roc_auc"]["mean"]
               for nm in MODELS}
    headline_th = max(sel_auc, key=sel_auc.get)
    metrics["time_holdout_headline_model"] = headline_th
    metrics["time_holdout_headline_model_rule"] = (
        f"highest mean ROC-AUC in {N_SPLITS}x{N_REPEATS} repeated stratified CV on the earliest "
        f"{n_tr_sel} (training-period) wafers only; no test-period label is used")
    metrics["time_holdout_headline_selection_cv_roc_auc"] = sel_auc
    th_curves = {}
    metrics["time_holdout_top10_sensors"] = {}
    for name in MODELS:
        r = time_holdout(name, X, y, None)
        r["roc_auc_95ci_bootstrap_test_wafers"] = bootstrap_auc_ci(r, y)
        th_curves[name] = (np.array(r["_tpr_grid"]), None)
        metrics["time_holdout"][name] = r
        r10 = time_holdout(name, X, y, 10)
        metrics["time_holdout_top10_infold"][name] = r10
        metrics["time_holdout_top10_sensors"][name] = {
            "fit_on_earliest_80pct_of_training_period": [f"S{j + 1:03d}" for j in r10["_orig"]],
            "selection_frequency_over_train_period_cv_top10": top10_selection_report(name, X, y),
        }
        print(f"time holdout {name}: AUC {r['roc_auc']:.3f}")

    chart_roc(curves_cv, th_curves, "nested", headline_th, OUT_DIR / "roc_recall_vs_false_alarm.png")
    chart_sensors(top, headline, OUT_DIR / "top15_sensors.png")
    chart_time(y, ts, OUT_DIR / "fail_rate_over_time.png", n_tr)

    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT_DIR / "metrics.json").write_text(json.dumps(strip(metrics), indent=2))
    print(f"done in {metrics['runtime_seconds']}s")


if __name__ == "__main__":
    main()
