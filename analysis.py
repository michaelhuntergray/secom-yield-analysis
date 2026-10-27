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
                                   class_weight="balanced", max_iter=2000),
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


def build_fixed(name):
    """Pipeline for a fixed column subset: no filter/select, same model grid."""
    make, grid = MODELS[name]
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("model", make())])
    return pipe, dict(grid)


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
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_t].max())


def fit_and_eval(pipe, grid, Xtr, ytr, Xte, yte, seed, orig_cols=None):
    """Tune on the training part only, set thresholds on training-part
    out-of-fold scores only, evaluate on the held-out part."""
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1, refit=True)
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    s_te = score(best, Xte)
    # thresholds: training-part out-of-fold scores of the tuned pipeline
    oof = cross_val_predict(clone(best), Xtr, ytr, method=method_for(best),
                            cv=StratifiedKFold(5, shuffle=True, random_state=seed + 1))
    if oof.ndim == 2:
        oof = oof[:, 1]
    neg = oof[ytr == 0]
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
    fpr, tpr, _ = roc_curve(yte, s_te)
    res["_tpr_grid"] = np.interp(FPR_GRID, fpr, tpr).tolist()
    # which original columns were used and how important were they
    if orig_cols is None:
        kept = np.flatnonzero(best.named_steps["filter"].support_)
        sel = best.named_steps["select"].get_support(indices=True)
        orig = kept[sel]
    else:
        orig = np.asarray(orig_cols)
    m = best.named_steps["model"]
    imp = np.abs(m.coef_.ravel()) if isinstance(m, LogisticRegression) else m.feature_importances_
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    res["_orig"], res["_imp"] = orig.tolist(), imp.tolist()
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
    n_tr = int(round(TRAIN_FRAC * len(y)))
    pipe, grid = build(name, k)
    res = fit_and_eval(pipe, grid, X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], seed)
    res["n_train"], res["n_train_fail"] = n_tr, int(y[:n_tr].sum())
    return res


def bootstrap_auc_ci(name, X, y, k, n_boot=1000):
    """95% bootstrap CI of the holdout ROC-AUC (resampling test wafers only)."""
    n_tr = int(round(TRAIN_FRAC * len(y)))
    pipe, grid = build(name, k)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc",
                      cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=1)
    gs.fit(X[:n_tr], y[:n_tr])
    s, yt = score(gs.best_estimator_, X[n_tr:]), y[n_tr:]
    rng = np.random.default_rng(SEED)
    aucs = []
    while len(aucs) < n_boot:
        i = rng.integers(0, len(yt), len(yt))
        if 0 < yt[i].sum() < len(i):
            aucs.append(roc_auc_score(yt[i], s[i]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def fixed_top10_holdout(name, X, y, seed=SEED):
    """Rank sensors on the training period only (selection frequency over a
    5x3 repeated CV of that period), then fit on those 10 sensors and test on
    the later period. No test-period data touches the ranking."""
    n_tr = int(round(TRAIN_FRAC * len(y)))
    Xtr, ytr = X[:n_tr], y[:n_tr]
    folds = run_cv(name, Xtr, ytr, k=None, n_repeats=3)
    top = sensor_ranking(folds, X.shape[1], top=10)
    cols = [t["col_index_1based"] - 1 for t in top]
    pipe, grid = build_fixed(name)
    res = fit_and_eval(pipe, grid, Xtr[:, cols], ytr, X[n_tr:][:, cols], y[n_tr:], seed,
                       orig_cols=cols)
    res["sensors"] = [t["sensor"] for t in top]
    return res


# ------------------------------------------------------------------ charts
COL = {"logreg_l1": "#1f77b4", "logreg_l2": "#2ca02c", "random_forest": "#d62728"}
LAB = {"logreg_l1": "Logistic L1", "logreg_l2": "Logistic L2", "random_forest": "Random forest"}


def chart_roc(cv_curves, th_curves, headline, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, title, curves in [
        (axes[0], f"Repeated stratified {N_SPLITS}-fold CV ({N_SPLITS}x{N_REPEATS}), mean of folds", cv_curves),
        (axes[1], "Time-ordered holdout (train earliest 70%, test latest 30%)", th_curves),
    ]:
        for name, (m, sd) in curves.items():
            lw = 2.6 if name == headline else 1.4
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
    fig.suptitle(f"Top 15 sensors, {LAB[headline]} model", fontsize=12)
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
            "threshold_trainthr": "score threshold at the target FPR of out-of-fold scores on the training part (applied unchanged to the test part)",
            "threshold_roc": "best recall with FPR <= target read off the test-part ROC curve (optimistic: oracle threshold)",
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
    headline = max(MODELS, key=lambda m: metrics["cv"][m]["roc_auc"]["mean"])
    metrics["headline_model"] = headline
    metrics["headline_model_rule"] = "highest mean CV ROC-AUC among the 3 models; all 3 are reported"

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
    th_curves = {}
    metrics["time_holdout_top10_fixed_from_train_period"] = {}
    for name in MODELS:
        r = time_holdout(name, X, y, None)
        r["roc_auc_95ci_bootstrap_test_wafers"] = bootstrap_auc_ci(name, X, y, None)
        th_curves[name] = (np.array(r["_tpr_grid"]), None)
        metrics["time_holdout"][name] = r
        metrics["time_holdout_top10_infold"][name] = time_holdout(name, X, y, 10)
        metrics["time_holdout_top10_fixed_from_train_period"][name] = fixed_top10_holdout(name, X, y)
        print(f"time holdout {name}: AUC {r['roc_auc']:.3f}")

    chart_roc(curves_cv, th_curves, headline, OUT_DIR / "roc_recall_vs_false_alarm.png")
    chart_sensors(top, headline, OUT_DIR / "top15_sensors.png")
    chart_time(y, ts, OUT_DIR / "fail_rate_over_time.png", n_tr)

    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT_DIR / "metrics.json").write_text(json.dumps(strip(metrics), indent=2))
    print(f"done in {metrics['runtime_seconds']}s")


if __name__ == "__main__":
    main()
