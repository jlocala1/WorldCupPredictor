"""Train, tune, calibrate, and evaluate models on features.csv.

Pipeline:
  1. Feature selection (FEATURE_FILTERS) + missingness indicator columns
  2. Median imputation on training data only (no leakage)
  3. Per-model training:
       - Logistic Regression (scaled features, class-balanced)
       - Random Forest        (no scaling, class-balanced sampling)
       - XGBoost              (no scaling, sample-weighted for class balance)
  4. Hyperparameter tuning per model — small grid search using val set
  5. Probability calibration — isotonic regression on top of best models
  6. Soft-voting ensemble of the three calibrated models
  7. Leave-one-group-out ablation on the best individual model
  8. Save artifacts + plots + JSON summaries for the report

Outputs:
  models/
    lr.pkl, rf.pkl, xgb.pkl       — best calibrated model per family
    lr_raw.pkl, rf_raw.pkl, ...   — uncalibrated versions (used by simulate.py)
    scaler.pkl                    — StandardScaler for LR
    fill_values.pkl               — per-column median fill values for inference
    feature_names.pkl             — ordered list of feature columns
    classes.pkl                   — [0, 1, 2]
    best_params.json              — chosen hyperparameters per model
    summary.json                  — val metrics per model
    ablation.json                 — leave-one-group-out results

  data/processed/
    cm_lr.png, cm_rf.png, cm_xgb.png        — confusion matrices
    fi_lr.png, fi_rf.png, fi_xgb.png        — feature importance plots
    ablation.png                            — leave-one-group-out chart

Run with: python src/models.py
Methodology details: see METHODOLOGY.md at project root.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless rendering for PNGs
import matplotlib.pyplot as plt  # noqa: E402
from itertools import product  # noqa: E402

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, confusion_matrix, f1_score, log_loss,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# SoftVoteEnsemble lives in its own module so pickle stores it with a stable
# qualified name (`ensemble.SoftVoteEnsemble`) regardless of how this script
# is launched. Defining the class inside models.py would pickle it as
# `__main__.SoftVoteEnsemble`, which then fails to unpickle from simulate.py.
from ensemble import SoftVoteEnsemble  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "data" / "processed"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"

# NOTE: `caps` is intentionally omitted. The full-grid ablation showed
# avg_caps was the only feature group whose REMOVAL slightly improved log
# loss (Δll = -0.0014) — i.e. it's redundant once squad_value is in the
# model and adds variance for no signal. Squad value already encodes
# "experienced players play for valuable clubs."
FEATURE_FILTERS = ["form_", "h2h_", "elo", "fifa_", "squad_", "top26_",
                   "avg_value", "_z", "neutral"]
TARGET_NAMES = ["away_win", "draw", "home_win"]
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted list of base feature columns matching FEATURE_FILTERS."""
    return sorted([c for c in df.columns if any(x in c for x in FEATURE_FILTERS)])


def prepare_features(df: pd.DataFrame):
    """Build train/val/test feature matrices with missingness indicators and median fill.

    Returns:
        X_train, X_val, X_test: imputed-but-unscaled DataFrames
        Y_train, Y_val, Y_test: integer labels
        feature_names: ordered list of all columns (base + indicators)
        fill_values:   dict[col -> fill value], for inference time
    """
    base_features = select_feature_columns(df)

    train_df = df[df["split"] == "train"].dropna(subset=["label"]).copy()
    val_df = df[df["split"] == "val"].dropna(subset=["label"]).copy()
    test_df = df[df["split"] == "test"].dropna(subset=["label"]).copy()

    X_train_raw = train_df[base_features].copy()
    X_val_raw = val_df[base_features].copy()
    X_test_raw = test_df[base_features].copy()
    Y_train = train_df["label"].astype(int).reset_index(drop=True)
    Y_val = val_df["label"].astype(int).reset_index(drop=True)
    Y_test = test_df["label"].astype(int).reset_index(drop=True)

    # Indicator columns for any feature that has a NaN in train, val, OR test
    indicator_cols: list[str] = []
    for col in base_features:
        if (X_train_raw[col].isna().any() or X_val_raw[col].isna().any()
                or X_test_raw[col].isna().any()):
            ind = f"{col}_missing"
            X_train_raw[ind] = X_train_raw[col].isna().astype(int)
            X_val_raw[ind] = X_val_raw[col].isna().astype(int)
            X_test_raw[ind] = X_test_raw[col].isna().astype(int)
            indicator_cols.append(ind)

    # Median imputation (compute medians on TRAIN ONLY to avoid leakage)
    medians = X_train_raw[base_features].median()
    X_train = X_train_raw.fillna(medians).reset_index(drop=True)
    X_val = X_val_raw.fillna(medians).reset_index(drop=True)
    X_test = X_test_raw.fillna(medians).reset_index(drop=True)

    feature_names = list(X_train.columns)
    fill_values = {c: float(medians[c]) for c in base_features}
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, feature_names, fill_values


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _score_for_selection(model, X_val_scaled_or_raw, Y_val) -> float:
    """Score function used in tuning: log loss + 0.5*macro_F1.

    Combines a probabilistic measure (log loss, lower better) with a class-balance
    measure (macro F1, higher better) so we don't pick configs that just predict
    home_win 95% of the time.
    """
    probs = model.predict_proba(X_val_scaled_or_raw)
    preds = probs.argmax(axis=1)
    ll = log_loss(Y_val, probs, labels=[0, 1, 2])
    f1m = f1_score(Y_val, preds, average="macro")
    # Lower is better (we minimise this). Want low log loss AND high F1.
    return ll - 0.5 * f1m


def tune_logistic_regression(X_train, Y_train, X_val, Y_val):
    """Grid-search LR over C (regularization strength). Scaler refit per try.

    Returns (best_lr, best_scaler, best_params, attempts).
    """
    print("\n=== Tuning Logistic Regression ===")
    grid = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    best_score = float("inf")
    best_model = None
    best_scaler = None
    best_params = None
    attempts = []
    for c in grid["C"]:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_va = scaler.transform(X_val)
        model = LogisticRegression(
            C=c, max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=RANDOM_STATE,
        )
        model.fit(X_tr, Y_train)
        score = _score_for_selection(model, X_va, Y_val)
        attempts.append({"C": c, "score": score,
                         "log_loss": log_loss(Y_val, model.predict_proba(X_va), labels=[0, 1, 2]),
                         "macro_f1": f1_score(Y_val, model.predict(X_va), average="macro")})
        print(f"  C={c:>7}  score={score:+.4f}  log_loss={attempts[-1]['log_loss']:.4f}  macro_f1={attempts[-1]['macro_f1']:.4f}")
        if score < best_score:
            best_score = score
            best_model = model
            best_scaler = scaler
            best_params = {"C": c}
    print(f"  best: {best_params}  score={best_score:+.4f}")
    return best_model, best_scaler, best_params, attempts


def tune_random_forest(X_train, Y_train, X_val, Y_val):
    """Grid-search RF over n_estimators / max_depth / min_samples_leaf."""
    print("\n=== Tuning Random Forest ===")
    # min_samples_leaf grid extended to 10 and 20 because the previous run
    # picked the most-regularized value (5) — typically a sign there's more
    # room to regularize.
    grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 15, 25],
        "min_samples_leaf": [1, 2, 5, 10, 20],
    }
    best_score = float("inf")
    best_model = None
    best_params = None
    attempts = []
    for ne, md, msl in product(grid["n_estimators"], grid["max_depth"], grid["min_samples_leaf"]):
        model = RandomForestClassifier(
            n_estimators=ne, max_depth=md, min_samples_leaf=msl,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
        )
        model.fit(X_train, Y_train)
        score = _score_for_selection(model, X_val, Y_val)
        ll = log_loss(Y_val, model.predict_proba(X_val), labels=[0, 1, 2])
        f1m = f1_score(Y_val, model.predict(X_val), average="macro")
        attempts.append({"n_estimators": ne, "max_depth": md, "min_samples_leaf": msl,
                         "score": score, "log_loss": ll, "macro_f1": f1m})
        print(f"  n_est={ne:>4}  max_depth={str(md):<5}  min_leaf={msl}  "
              f"score={score:+.4f}  log_loss={ll:.4f}  macro_f1={f1m:.4f}")
        if score < best_score:
            best_score = score
            best_model = model
            best_params = {"n_estimators": ne, "max_depth": md, "min_samples_leaf": msl}
    print(f"  best: {best_params}  score={best_score:+.4f}")
    return best_model, best_params, attempts


def tune_xgboost(X_train, Y_train, X_val, Y_val):
    """Grid-search XGBoost over n_estimators / max_depth / learning_rate."""
    print("\n=== Tuning XGBoost ===")
    if not HAS_XGB:
        return None, None, []
    grid = {
        "n_estimators": [300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
    }
    sample_w = compute_sample_weight(class_weight="balanced", y=Y_train)
    best_score = float("inf")
    best_model = None
    best_params = None
    attempts = []
    for ne, md, lr in product(grid["n_estimators"], grid["max_depth"], grid["learning_rate"]):
        model = XGBClassifier(
            n_estimators=ne, max_depth=md, learning_rate=lr,
            subsample=0.85, colsample_bytree=0.85,
            objective="multi:softprob", num_class=3,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric="mlogloss",
        )
        model.fit(X_train, Y_train, sample_weight=sample_w)
        score = _score_for_selection(model, X_val, Y_val)
        ll = log_loss(Y_val, model.predict_proba(X_val), labels=[0, 1, 2])
        f1m = f1_score(Y_val, model.predict(X_val), average="macro")
        attempts.append({"n_estimators": ne, "max_depth": md, "learning_rate": lr,
                         "score": score, "log_loss": ll, "macro_f1": f1m})
        print(f"  n_est={ne:>4}  max_depth={md}  lr={lr:.2f}  "
              f"score={score:+.4f}  log_loss={ll:.4f}  macro_f1={f1m:.4f}")
        if score < best_score:
            best_score = score
            best_model = model
            best_params = {"n_estimators": ne, "max_depth": md, "learning_rate": lr}
    print(f"  best: {best_params}  score={best_score:+.4f}")
    return best_model, best_params, attempts


# ---------------------------------------------------------------------------
# Calibration + Ensemble
# ---------------------------------------------------------------------------

def calibrate(model, X_train, Y_train, method: str = "isotonic"):
    """Wrap a fitted model with a calibration layer fit via cross-validated
    isotonic regression on the training data. Improves probability quality
    (log loss) without changing accuracy much.
    """
    cal = CalibratedClassifierCV(model, method=method, cv=5)
    cal.fit(X_train, Y_train)
    return cal


def predict_proba_ensemble(probs_list, weights=None):
    """Soft-voting average of an iterable of (n_samples, 3) probability arrays."""
    arr = np.stack(probs_list, axis=0)
    if weights is None:
        return arr.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return (arr * w[:, None, None]).sum(axis=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def predict_proba_lr(model, scaler, X):
    return model.predict_proba(scaler.transform(X))


def evaluate(name: str, probs: np.ndarray, preds: np.ndarray, Y_val: pd.Series) -> dict:
    metrics = {
        "name": name,
        "accuracy": float(accuracy_score(Y_val, preds)),
        "macro_f1": float(f1_score(Y_val, preds, average="macro")),
        "log_loss": float(log_loss(Y_val, probs, labels=[0, 1, 2])),
    }
    print(f"\n=== {name} ===")
    print(f"  accuracy = {metrics['accuracy']:.4f}")
    print(f"  macro F1 = {metrics['macro_f1']:.4f}")
    print(f"  log loss = {metrics['log_loss']:.4f}")
    print(classification_report(Y_val, preds, target_names=TARGET_NAMES, digits=3, zero_division=0))
    return metrics


def plot_confusion_matrix(preds: np.ndarray, Y_val: pd.Series, name: str, path: Path) -> None:
    cm = confusion_matrix(Y_val, preds, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(TARGET_NAMES); ax.set_yticklabels(TARGET_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    cmax = cm.max() or 1
    for i in range(3):
        for j in range(3):
            colour = "white" if cm[i, j] > cmax * 0.6 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=colour)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_importance(importances: np.ndarray, feature_names: list[str],
                            name: str, path: Path, top: int = 20) -> None:
    if importances is None or len(importances) == 0:
        return
    idx = np.argsort(np.abs(importances))[-top:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), np.abs(importances[idx]))
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top} Features — {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

# Each group lists the substring filters used to identify that group's feature
# columns. We strip both the base column and any matching `*_missing` indicator
# when ablating, so the model genuinely loses access to that signal.
ABLATION_GROUPS: dict[str, list[str]] = {
    "form":         ["form_"],
    "h2h":          ["h2h_"],
    "elo":          ["elo"],
    "fifa_rank":    ["fifa_"],
    "squad_value":  ["squad_value", "top26_value", "avg_value", "squad_size"],
    "caps":         ["avg_caps"],
    "z_scores":     ["_z"],
    "neutral":      ["neutral"],
}


def _columns_for_group(feature_names: list[str], filters: list[str]) -> list[str]:
    """Return all columns (base + any *_missing indicators) matching the group filters."""
    return [c for c in feature_names if any(f in c for f in filters)]


def run_ablation(X_train: pd.DataFrame, Y_train, X_val: pd.DataFrame, Y_val,
                 feature_names: list[str], best_xgb_params: dict | None) -> dict:
    """Leave-one-group-out ablation on XGBoost (or RF if XGB unavailable).

    For each feature group, retrain a fresh model on X_train minus that group's
    columns and report val accuracy / macro F1 / log loss. The full-feature
    baseline is included for comparison. We use the best XGB hyperparams found
    in tuning so we're isolating the *features*' contribution, not interaction
    with regularization.
    """
    print("\n=== Leave-one-group-out ablation ===")

    def fit(cols: list[str]):
        if HAS_XGB and best_xgb_params is not None:
            sample_w = compute_sample_weight(class_weight="balanced", y=Y_train)
            m = XGBClassifier(
                **best_xgb_params,
                subsample=0.85, colsample_bytree=0.85,
                objective="multi:softprob", num_class=3,
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric="mlogloss",
            )
            m.fit(X_train[cols], Y_train, sample_weight=sample_w)
        else:
            m = RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=2,
                class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
            )
            m.fit(X_train[cols], Y_train)
        probs = m.predict_proba(X_val[cols])
        preds = probs.argmax(axis=1)
        return {
            "accuracy": float(accuracy_score(Y_val, preds)),
            "macro_f1": float(f1_score(Y_val, preds, average="macro")),
            "log_loss": float(log_loss(Y_val, probs, labels=[0, 1, 2])),
            "n_features": len(cols),
        }

    results: dict[str, dict] = {"all_features": fit(list(feature_names))}
    base = results["all_features"]
    print(f"  {'all_features':<16} acc={base['accuracy']:.4f}  f1={base['macro_f1']:.4f}  "
          f"ll={base['log_loss']:.4f}  ({base['n_features']} feats)")
    for group, filters in ABLATION_GROUPS.items():
        drop = set(_columns_for_group(feature_names, filters))
        keep = [c for c in feature_names if c not in drop]
        if not drop or len(keep) == len(feature_names):
            print(f"  {group:<16} (no matching features — skipped)")
            continue
        m = fit(keep)
        m["dropped"] = sorted(drop)
        results[f"no_{group}"] = m
        delta_ll = m["log_loss"] - base["log_loss"]
        delta_f1 = m["macro_f1"] - base["macro_f1"]
        print(f"  no_{group:<13} acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}  "
              f"ll={m['log_loss']:.4f}  Δll={delta_ll:+.4f}  Δf1={delta_f1:+.4f}  "
              f"(-{len(drop)} feats)")
    return results


def plot_ablation(results: dict, path: Path) -> None:
    """Bar chart of Δlog_loss vs full-feature baseline for each group dropped.
    Positive Δlog_loss = removing that group made the model WORSE = group helps.
    """
    base_ll = results["all_features"]["log_loss"]
    rows = [(k.removeprefix("no_"), v["log_loss"] - base_ll)
            for k, v in results.items() if k != "all_features"]
    rows.sort(key=lambda r: r[1])
    if not rows:
        return
    names, deltas = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = ["tab:green" if d > 0 else "tab:red" for d in deltas]
    ax.barh(range(len(names)), deltas, color=colours)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ log loss vs full-feature baseline\n(positive = group helps; negative = group hurts)")
    ax.set_title("Leave-one-group-out ablation (XGBoost)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Backwards-compat helpers (called by simulate.py the teammate is writing)
# ---------------------------------------------------------------------------

def prepare_data(df):
    """Legacy interface kept so existing simulate.py code doesn't break.

    Returns scaled features (LR-style), used by simulators that load
    imputer.pkl + scaler.pkl + lr.pkl. New code should prefer
    `prepare_features()` which returns the unscaled imputed frame.
    """
    X_train, X_val, _Xt, Y_train, Y_val, _Yt, feature_names, fill_values = prepare_features(df)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    # `imputer` is conceptual now (we did per-column median fill manually)
    # — return fill_values dict for any callers that want to apply the same fills.
    return X_train_scaled, X_val_scaled, Y_train, Y_val, feature_names, fill_values, scaler


# ---------------------------------------------------------------------------
# Main: train all three, save everything
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    X_train, X_val, X_test, Y_train, Y_val, Y_test, feature_names, fill_values = prepare_features(df)
    print(f"Train rows: {len(X_train):,}   Val rows: {len(X_val):,}   Test rows: {len(X_test):,}")
    print(f"Total features: {len(feature_names)} "
          f"({len([c for c in feature_names if not c.endswith('_missing')])} base "
          f"+ {len([c for c in feature_names if c.endswith('_missing')])} missingness flags)")

    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    best_params: dict[str, dict] = {}
    tuning_log: dict[str, list[dict]] = {}

    # --- Logistic Regression -------------------------------------------------
    lr_raw, scaler, lr_params, lr_attempts = tune_logistic_regression(
        X_train, Y_train, X_val, Y_val)
    best_params["logistic_regression"] = lr_params
    tuning_log["logistic_regression"] = lr_attempts
    # Calibrate using a fresh LR fitted inside the calibrator on scaled features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    lr_for_cal = LogisticRegression(C=lr_params["C"], max_iter=2000, solver="lbfgs",
                                    class_weight="balanced", random_state=RANDOM_STATE)
    lr_cal = calibrate(lr_for_cal, X_train_scaled, Y_train, method="isotonic")
    lr_probs = lr_cal.predict_proba(X_val_scaled)
    lr_preds = lr_probs.argmax(axis=1)
    metrics = evaluate("Logistic Regression", lr_probs, lr_preds, Y_val)
    all_metrics.append(metrics)
    plot_confusion_matrix(lr_preds, Y_val, "Logistic Regression", FIGURES_DIR / "cm_lr.png")
    plot_feature_importance(np.abs(lr_raw.coef_).mean(axis=0), feature_names,
                            "Logistic Regression", FIGURES_DIR / "fi_lr.png")
    joblib.dump(lr_cal, MODELS_DIR / "lr.pkl")
    joblib.dump(lr_raw, MODELS_DIR / "lr_raw.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # --- Random Forest -------------------------------------------------------
    rf_raw, rf_params, rf_attempts = tune_random_forest(X_train, Y_train, X_val, Y_val)
    best_params["random_forest"] = rf_params
    tuning_log["random_forest"] = rf_attempts
    rf_for_cal = RandomForestClassifier(
        **rf_params, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf_cal = calibrate(rf_for_cal, X_train, Y_train, method="isotonic")
    rf_probs = rf_cal.predict_proba(X_val)
    rf_preds = rf_probs.argmax(axis=1)
    metrics = evaluate("Random Forest", rf_probs, rf_preds, Y_val)
    all_metrics.append(metrics)
    plot_confusion_matrix(rf_preds, Y_val, "Random Forest", FIGURES_DIR / "cm_rf.png")
    plot_feature_importance(rf_raw.feature_importances_, feature_names,
                            "Random Forest", FIGURES_DIR / "fi_rf.png")
    joblib.dump(rf_cal, MODELS_DIR / "rf.pkl")
    joblib.dump(rf_raw, MODELS_DIR / "rf_raw.pkl")

    # --- XGBoost -------------------------------------------------------------
    xgb_probs = None
    xgb_raw = None
    xgb_params = None
    if HAS_XGB:
        xgb_raw, xgb_params, xgb_attempts = tune_xgboost(X_train, Y_train, X_val, Y_val)
        best_params["xgboost"] = xgb_params
        tuning_log["xgboost"] = xgb_attempts
        # Calibrate a fresh XGB on the same training data
        sample_w = compute_sample_weight(class_weight="balanced", y=Y_train)
        xgb_for_cal = XGBClassifier(
            **xgb_params, subsample=0.85, colsample_bytree=0.85,
            objective="multi:softprob", num_class=3,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric="mlogloss",
        )
        xgb_for_cal.fit(X_train, Y_train, sample_weight=sample_w)
        # CalibratedClassifierCV with cv="prefit" uses a holdout — but we want
        # full-train fit, so use 5-fold cv calibration on a fresh estimator wrapper.
        xgb_cal = CalibratedClassifierCV(
            XGBClassifier(
                **xgb_params, subsample=0.85, colsample_bytree=0.85,
                objective="multi:softprob", num_class=3,
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric="mlogloss",
            ),
            method="isotonic", cv=5,
        )
        xgb_cal.fit(X_train, Y_train, sample_weight=sample_w)
        xgb_probs = xgb_cal.predict_proba(X_val)
        xgb_preds = xgb_probs.argmax(axis=1)
        metrics = evaluate("XGBoost", xgb_probs, xgb_preds, Y_val)
        all_metrics.append(metrics)
        plot_confusion_matrix(xgb_preds, Y_val, "XGBoost", FIGURES_DIR / "cm_xgb.png")
        plot_feature_importance(xgb_raw.feature_importances_, feature_names,
                                "XGBoost", FIGURES_DIR / "fi_xgb.png")
        joblib.dump(xgb_cal, MODELS_DIR / "xgb.pkl")
        joblib.dump(xgb_raw, MODELS_DIR / "xgb_raw.pkl")
    else:
        print("\nXGBoost not installed — skipping (`pip install xgboost`).")

    # --- Soft-voting ensemble ------------------------------------------------
    probs_list = [lr_probs, rf_probs]
    if xgb_probs is not None:
        probs_list.append(xgb_probs)
    ens_probs = predict_proba_ensemble(probs_list)
    ens_preds = ens_probs.argmax(axis=1)
    metrics = evaluate("Ensemble (soft-vote)", ens_probs, ens_preds, Y_val)
    all_metrics.append(metrics)
    plot_confusion_matrix(ens_preds, Y_val, "Ensemble", FIGURES_DIR / "cm_ensemble.png")

    # Save the ensemble as a single picklable object so simulate.py can load
    # it like any other model: ensemble = joblib.load("models/ensemble.pkl");
    # ensemble.predict_proba(X)  # → (n, 3) array averaged over LR + RF + XGB
    ensemble_model = SoftVoteEnsemble(
        lr=lr_cal, rf=rf_cal, xgb=xgb_cal if HAS_XGB else None, scaler=scaler,
    )
    joblib.dump(ensemble_model, MODELS_DIR / "ensemble.pkl")

    # --- Held-out test evaluation -------------------------------------------
    # Test split = 2025-2026-03 matches, never seen during training OR tuning.
    # This is the honest "out-of-sample" number for the report — comparing
    # against val tells us whether tuning over 35+ configs on val produced
    # an optimistic estimate or a robust one (low gap = robust).
    print("\n=== Held-out test set (2025-2026-03) ===")
    X_test_scaled = scaler.transform(X_test)
    test_lr_probs = lr_cal.predict_proba(X_test_scaled)
    test_rf_probs = rf_cal.predict_proba(X_test)
    test_probs_list = [test_lr_probs, test_rf_probs]
    if xgb_probs is not None:
        test_xgb_probs = xgb_cal.predict_proba(X_test)
        test_probs_list.append(test_xgb_probs)
    test_ens_probs = predict_proba_ensemble(test_probs_list)

    test_metrics: list[dict] = []
    for name, probs in [("Logistic Regression", test_lr_probs),
                        ("Random Forest", test_rf_probs),
                        *( [("XGBoost", test_xgb_probs)] if xgb_probs is not None else [] ),
                        ("Ensemble (soft-vote)", test_ens_probs)]:
        preds = probs.argmax(axis=1)
        m = evaluate(f"TEST — {name}", probs, preds, Y_test)
        test_metrics.append(m)
    plot_confusion_matrix(test_ens_probs.argmax(axis=1), Y_test, "Ensemble (test)",
                          FIGURES_DIR / "cm_ensemble_test.png")

    # --- Save shared artifacts ----------------------------------------------
    joblib.dump(fill_values, MODELS_DIR / "fill_values.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    joblib.dump([0, 1, 2], MODELS_DIR / "classes.pkl")

    # --- Side-by-side comparison --------------------------------------------
    print("\n=== Model Comparison (val vs test, after calibration) ===")
    print(f"{'Model':<22}  {'val_acc':>8} {'val_f1':>8} {'val_ll':>8}  "
          f"{'test_acc':>8} {'test_f1':>8} {'test_ll':>8}  {'Δll':>7}")
    test_by_name = {m["name"].removeprefix("TEST — "): m for m in test_metrics}
    for m in all_metrics:
        t = test_by_name.get(m["name"], {})
        delta_ll = t.get("log_loss", float("nan")) - m["log_loss"]
        print(f"{m['name']:<22}  {m['accuracy']:>8.4f} {m['macro_f1']:>8.4f} {m['log_loss']:>8.4f}  "
              f"{t.get('accuracy', float('nan')):>8.4f} {t.get('macro_f1', float('nan')):>8.4f} "
              f"{t.get('log_loss', float('nan')):>8.4f}  {delta_ll:>+7.4f}")

    summary = {
        "val": {m["name"]: {k: m[k] for k in ("accuracy", "macro_f1", "log_loss")}
                for m in all_metrics},
        "test": {m["name"].removeprefix("TEST — "): {k: m[k] for k in ("accuracy", "macro_f1", "log_loss")}
                 for m in test_metrics},
    }
    with (MODELS_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (MODELS_DIR / "best_params.json").open("w") as f:
        json.dump(best_params, f, indent=2, default=str)
    with (MODELS_DIR / "tuning_log.json").open("w") as f:
        json.dump(tuning_log, f, indent=2, default=str)

    # --- Ablation study ------------------------------------------------------
    ablation = run_ablation(X_train, Y_train, X_val, Y_val, feature_names, xgb_params)
    with (MODELS_DIR / "ablation.json").open("w") as f:
        json.dump(ablation, f, indent=2, default=str)
    plot_ablation(ablation, FIGURES_DIR / "ablation.png")

    print(f"\nSaved {len(all_metrics)} model(s), {len(ablation)-1} ablation runs, plots.")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Plots:  {FIGURES_DIR}")


if __name__ == "__main__":
    main()
