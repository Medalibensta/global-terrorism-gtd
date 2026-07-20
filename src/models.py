"""
Predictive modelling on GTD incidents (brief step 5).

Two complementary tasks from the same incident features (region, attack type,
target type, weapon type, suicide flag, year):

    * CLASSIFICATION — will an incident be lethal (nkill > 0)?  ~46% positive,
      so accuracy is meaningful but we also report ROC-AUC / F1.
      Models: logistic regression (baseline) vs gradient boosting.

    * REGRESSION — how severe (deaths + 0.5·wounded), on log1p scale to tame the
      heavy tail. Models: linear regression vs gradient boosting. Reported in
      log-RMSE and, back-transformed, MAE in casualties.

A strictly TEMPORAL split (train <=2013, test >=2014) is used instead of a random
split: predicting the future from the past is the only honest evaluation, and it
also stresses generalisation to the post-2013 surge.

Run:
    python src/models.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import RANDOM_STATE
from preprocessing import load_clean

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

CAT_FEATURES = ["region", "attack_type", "target_type", "weapon_type"]
NUM_FEATURES = ["suicide", "year"]
SPLIT_YEAR = 2014


def _prep() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_clean()
    df = df[df.year != 1993]  # missing year
    train = df[df.year < SPLIT_YEAR]
    test = df[df.year >= SPLIT_YEAR]
    return train, test


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ])


def run_classification() -> pd.DataFrame:
    train, test = _prep()
    Xtr, ytr = train[CAT_FEATURES + NUM_FEATURES], train["lethal"]
    Xte, yte = test[CAT_FEATURES + NUM_FEATURES], test["lethal"]

    models = {
        "LogReg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE),
    }
    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("prep", _preprocessor()), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(yte, pred),
            "roc_auc": roc_auc_score(yte, proba),
            "f1": f1_score(yte, pred),
        })
        print(f"[clf] {name}: acc={rows[-1]['accuracy']:.3f} "
              f"auc={rows[-1]['roc_auc']:.3f} f1={rows[-1]['f1']:.3f}")
    return pd.DataFrame(rows)


def run_regression() -> pd.DataFrame:
    train, test = _prep()
    ytr = np.log1p(train["severity"])
    yte_log = np.log1p(test["severity"])
    Xtr = train[CAT_FEATURES + NUM_FEATURES]
    Xte = test[CAT_FEATURES + NUM_FEATURES]

    models = {
        "LinearRegression": LinearRegression(),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    rows, preds = [], {}
    for name, reg in models.items():
        pipe = Pipeline([("prep", _preprocessor()), ("reg", reg)])
        pipe.fit(Xtr, ytr)
        pred_log = pipe.predict(Xte)
        preds[name] = pred_log
        rmse_log = float(np.sqrt(mean_squared_error(yte_log, pred_log)))
        mae_cas = float(mean_absolute_error(
            np.expm1(yte_log), np.clip(np.expm1(pred_log), 0, None)))
        rows.append({
            "model": name,
            "rmse_log": rmse_log,
            "mae_casualties": mae_cas,
            "r2_log": r2_score(yte_log, pred_log),
        })
        print(f"[reg] {name}: rmse_log={rmse_log:.3f} "
              f"mae={mae_cas:.2f} r2={rows[-1]['r2_log']:.3f}")

    # Predicted vs actual scatter for the better model.
    best = max(rows, key=lambda r: r["r2_log"])["model"]
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(np.expm1(yte_log), np.clip(np.expm1(preds[best]), 0, None),
                s=5, alpha=0.2)
    lim = 60
    plt.plot([0, lim], [0, lim], "r--", alpha=0.7)
    plt.xlim(0, lim); plt.ylim(0, lim)
    plt.xlabel("Sévérité réelle"); plt.ylabel("Sévérité prédite")
    plt.title(f"Prédiction de sévérité — {best} (test >= {SPLIT_YEAR})")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "severity_pred_vs_actual.png", dpi=130)
    plt.close()
    return pd.DataFrame(rows)


def run(save: bool = True) -> None:
    print("=== Classification: incident lethal? (temporal split) ===")
    clf = run_classification()
    print("\n=== Regression: incident severity (log1p) ===")
    reg = run_regression()
    if save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        clf.to_csv(REPORT_DIR / "classification_metrics.csv", index=False)
        reg.to_csv(REPORT_DIR / "regression_metrics.csv", index=False)
        print(f"\n[models] saved metrics to {REPORT_DIR}")


if __name__ == "__main__":
    run()
