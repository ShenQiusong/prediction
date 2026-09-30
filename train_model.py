# -*- coding: utf-8 -*-
"""Train the RF model once and save model + cutoff + SHAP explainer to disk,
so app.py can load them instantly instead of re-training at startup.

Run:  python train_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

PREDICTORS = [
    "Hospitalizations_last_year",
    "Dialysis_vintage",
    "CRP",
    "TIBC",
    "Total_cholesterol",
]


def main():
    df = pd.read_csv(DATA / "clean_data.csv")
    split = pd.read_csv(DATA / "split_index.csv").sort_values("row_index")
    df = df.copy()
    df["Set"] = split["Set"].to_numpy()
    train = df[df["Set"] == "Training"]

    Xtr, ytr = train[PREDICTORS], train["Malnutrition"].astype(int)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_features=2,
        min_samples_leaf=10,
        random_state=20260927,
        n_jobs=-1,
    )
    rf.fit(Xtr, ytr)

    ptr = rf.predict_proba(Xtr)[:, 1]
    fpr, tpr, thr = roc_curve(ytr, ptr)
    cutoff = float(thr[np.argmax(tpr - fpr)])

    explainer = shap.Explainer(rf)

    joblib.dump({"model": rf, "cutoff": cutoff}, DATA / "model_rf.pkl")
    joblib.dump(explainer, DATA / "explainer.pkl")
    print(f"Saved. cutoff = {cutoff:.4f}")


if __name__ == "__main__":
    main()
