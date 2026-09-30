# Malnutrition Risk Prediction in MHD Patients

A single-page Streamlit app for interactive malnutrition risk prediction in
maintenance hemodialysis (MHD) patients, with a per-patient SHAP force plot.

- **Model**: Random Forest (mtry = 2, nodesize = 10), trained on 420 MHD patients
- **Predictors**: Hospitalizations in the past year, Dialysis vintage, CRP, TIBC, Total cholesterol
- **Cutoff**: optimal threshold by Youden index on the training set

The app loads pre-trained artifacts (`data/model_rf.pkl`, `data/explainer.pkl`)
for fast startup. To re-train from scratch, run `python train_model.py`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io/ and sign in with GitHub.
2. Click **New app**, select the repository `ShenQiusong/prediction`,
   branch `main`, and main file path `app.py`.
3. Click **Deploy**.
