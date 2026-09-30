# -*- coding: utf-8 -*-
"""
Malnutrition Risk Prediction in Maintenance Hemodialysis Patients
Single-page Streamlit app: interactive risk prediction + per-patient SHAP force plot.

Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

PREDICTORS = [
    "Hospitalizations_last_year",
    "Dialysis_vintage",
    "CRP",
    "TIBC",
    "Total_cholesterol",
]

st.set_page_config(
    page_title="Malnutrition Risk Prediction in MHD Patients",
    page_icon="🩸",
    layout="centered",
)


# ---------------------------------------------------------------- model
@st.cache_data
def load_data():
    df = pd.read_csv(DATA / "clean_data.csv")
    split = pd.read_csv(DATA / "split_index.csv").sort_values("row_index")
    df = df.copy()
    df["Set"] = split["Set"].to_numpy()
    return df


@st.cache_resource
def train_rf():
    """Re-train the best model (RF) in Python, mirroring the R setup:
    mtry = 2  -> max_features = 2
    nodesize = 10 -> min_samples_leaf = 10
    Same 7:3 stratified split (split_index.csv), same 5 predictors.
    """
    df = load_data()
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

    # Optimal cutoff: Youden index on the training set
    ptr = rf.predict_proba(Xtr)[:, 1]
    fpr, tpr, thr = roc_curve(ytr, ptr)
    cutoff = float(thr[np.argmax(tpr - fpr)])

    return rf, cutoff


@st.cache_resource
def get_explainer(_rf):
    return shap.Explainer(_rf)


# ---------------------------------------------------------------- plots
def gauge(prob, cutoff):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Predicted probability of malnutrition", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, cutoff * 100], "color": "#d4edda"},
                    {"range": [cutoff * 100, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": cutoff * 100,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=20))
    return fig


def render_force_plot(explainer, x):
    """Official SHAP force plot for the current patient (class = malnutrition)."""
    sv = explainer(x)
    fp = shap.plots.force(
        explainer.expected_value[1],
        sv.values[0, :, 1],
        x.iloc[0],
        feature_names=PREDICTORS,
    )
    components.html(shap.getjs() + fp.html(), height=170, scrolling=False)


# ---------------------------------------------------------------- page
st.title("Malnutrition Risk Prediction in MHD Patients")
st.caption(
    "Random Forest model (mtry = 2, nodesize = 10) trained on 420 maintenance "
    "hemodialysis patients. Enter patient characteristics below."
)

rf, cutoff = train_rf()
explainer = get_explainer(rf)

st.subheader("Patient inputs")
col1, col2 = st.columns(2)
with col1:
    hosp = st.number_input("Hospitalizations in the past year (times)", 0, 3, 1, 1)
    vintage = st.number_input("Dialysis vintage (months)", 1, 280, 48, 1)
    crp = st.number_input("CRP (mg/L)", 0.0, 310.0, 4.5, 0.1, format="%.2f")
with col2:
    tibc = st.number_input("Total iron binding capacity, TIBC (μmol/L)", 10.0, 75.0, 34.0, 0.1, format="%.2f")
    tc = st.number_input("Total cholesterol (mmol/L)", 1.0, 8.0, 3.9, 0.01, format="%.2f")

x = pd.DataFrame([[hosp, vintage, crp, tibc, tc]], columns=PREDICTORS)
prob = float(rf.predict_proba(x)[0, 1])

st.subheader("Prediction")
st.plotly_chart(gauge(prob, cutoff), width="stretch")

if prob >= cutoff:
    st.error(f"**High risk** — predicted probability {prob:.1%} ≥ cutoff {cutoff:.1%}")
else:
    st.success(f"**Low risk** — predicted probability {prob:.1%} < cutoff {cutoff:.1%}")

st.subheader("Why this prediction? — SHAP force plot")
st.caption(
    "Red features push the prediction towards higher risk, blue features towards "
    "lower risk. The plot starts from the base probability (average prediction on "
    "the training set) and arrives at this patient's predicted probability."
)
render_force_plot(explainer, x)
