# -*- coding: utf-8 -*-
"""
Malnutrition Risk Prediction in Maintenance Hemodialysis Patients
Single-page Streamlit app: interactive risk prediction + per-patient SHAP force plot.

Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import streamlit.components.v1 as components

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
@st.cache_resource
def load_model():
    """Load the pre-trained RF model, cutoff and SHAP explainer.
    Artifacts are produced offline by train_model.py (mtry = 2, nodesize = 10,
    same 7:3 stratified split, same 5 predictors).
    """
    bundle = joblib.load(DATA / "model_rf.pkl")
    explainer = joblib.load(DATA / "explainer.pkl")
    return bundle["model"], bundle["cutoff"], explainer


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

rf, cutoff, explainer = load_model()

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
