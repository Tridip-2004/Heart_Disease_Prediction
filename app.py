# pip install streamlit joblib scikit-learn xgboost lightgbm imbalanced-learn

import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_artifacts():
    model         = joblib.load('best_model.pkl')
    scaler        = joblib.load('scaler.pkl')
    rfe           = joblib.load('rfe_selector.pkl')
    threshold     = joblib.load('threshold.pkl')
    encoding_maps = joblib.load('encoding_maps.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, rfe, threshold, encoding_maps, feature_names

model, scaler, rfe, threshold, encoding_maps, feature_names = load_artifacts()

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction")
st.markdown("Fill in the patient details below and click **Predict** to assess heart disease risk.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age             = st.number_input("Age (years)", min_value=1, max_value=120, value=55)
    sex             = st.selectbox("Sex", options=['M', 'F'], format_func=lambda x: 'Male' if x == 'M' else 'Female')
    chest_pain      = st.selectbox("Chest Pain Type", options=list(encoding_maps['ChestPainType'].keys()))
    resting_bp      = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
    cholesterol     = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=700, value=200)
    fasting_bs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=['Yes', 'No'])
    resting_ecg     = st.selectbox("Resting ECG Result", options=list(encoding_maps['RestingECG'].keys()))

with col2:
    max_hr          = st.number_input("Max Heart Rate Achieved (bpm)", min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", options=['Yes', 'No'])
    oldpeak         = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope           = st.selectbox("Slope of ST Segment", options=list(encoding_maps['SlopeOfST_Segment'].keys()))
    vessels         = st.selectbox("Major Vessels Colored", options=[0, 1, 2, 3],
                                    format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}")
    thalassemia     = st.selectbox("Thalassemia", options=list(encoding_maps['Thalassemia'].keys()))

st.divider()

if st.button("🔍 Predict", use_container_width=True, type="primary"):

    input_data = {
        'Age'                  : age,
        'Sex'                  : encoding_maps['Sex'][sex],
        'ChestPainType'        : encoding_maps['ChestPainType'][chest_pain],
        'RestingBloodPressure' : resting_bp,
        'Cholesterol'          : cholesterol,
        'FastingBloodSugar'    : encoding_maps['FastingBloodSugar'][fasting_bs],
        'RestingECG'           : encoding_maps['RestingECG'][resting_ecg],
        'MaxHeartRate'         : max_hr,
        'ExerciseInducedAngina': encoding_maps['ExerciseInducedAngina'][exercise_angina],
        'Oldpeak'              : oldpeak,
        'SlopeOfST_Segment'    : encoding_maps['SlopeOfST_Segment'][slope],
        'MajorVesselsColored'  : vessels,
        'Thalassemia'          : encoding_maps['Thalassemia'][thalassemia]
    }

    df_input = pd.DataFrame([input_data])
    df_input['Age_MaxHR_Ratio']  = df_input['Age'] / (df_input['MaxHeartRate'] + 1)
    df_input['BP_Chol_Ratio']    = df_input['RestingBloodPressure'] / (df_input['Cholesterol'] + 1)
    df_input['Age_Oldpeak']      = df_input['Age'] * df_input['Oldpeak']
    df_input['MaxHR_Oldpeak']    = df_input['MaxHeartRate'] / (df_input['Oldpeak'] + 1)
    df_input['ChestPain_Angina'] = df_input['ChestPainType'] * df_input['ExerciseInducedAngina']
    df_input['Slope_Oldpeak']    = df_input['SlopeOfST_Segment'] * df_input['Oldpeak']
    df_input['Age_Bin']          = pd.cut(df_input['Age'], bins=[0,40,50,60,70,100], labels=[0,1,2,3,4]).astype(int)
    df_input['MaxHR_Bin']        = pd.cut(df_input['MaxHeartRate'], bins=[0,100,130,160,220], labels=[0,1,2,3]).astype(int)
    df_input = df_input[feature_names]

    scaled     = scaler.transform(df_input)
    selected   = rfe.transform(scaled)
    prob       = model.predict_proba(selected)[0][1]
    prediction = int(prob >= threshold)

    # ── Result ───────────────────────────────────────────────
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ **Heart Disease Detected**")
    else:
        st.success("✅ **No Heart Disease Detected**")

    st.markdown(f"### Risk Probability: `{prob*100:.1f}%`")
    st.progress(float(prob))

    if prob < 0.3:
        st.markdown("**Risk Level:** 🟢 Low Risk")
    elif prob < 0.6:
        st.markdown("**Risk Level:** 🟡 Moderate Risk")
    else:
        st.markdown("**Risk Level:** 🔴 High Risk")

    st.caption(f"Decision threshold: {threshold:.2f} | Probability: {prob:.4f}")

    # ── Tips (only when heart disease detected) ──────────────
    if prediction == 1:
        st.divider()
        st.subheader("📋 Recommended Care Tips")
        st.markdown("Based on the prediction, here are important steps this patient should follow:")

        tips = {
            "🥗 Diet & Nutrition": [
                "Reduce saturated fats and trans fats",
                "Eat more fruits, vegetables, and whole grains",
                "Limit salt intake to under 1500 mg/day",
                "Avoid processed and fried foods",
            ],
            "🏃 Physical Activity": [
                "30 minutes of moderate exercise, 5 days/week",
                "Prefer walking, swimming, or cycling",
                "Avoid sudden high-intensity exertion",
                "Consult doctor before starting any gym routine",
            ],
            "💊 Medication & Monitoring": [
                "Take all prescribed medications regularly",
                "Monitor blood pressure daily at home",
                "Check cholesterol levels every 3 months",
                "Track resting heart rate each morning",
            ],
            "🚭 Lifestyle Changes": [
                "Quit smoking completely",
                "Limit alcohol to 1 drink/day or less",
                "Manage stress with meditation or yoga",
                "Get 7–8 hours of quality sleep per night",
            ],
            "🏥 Doctor Follow-ups": [
                "Visit cardiologist every 3–6 months",
                "Get ECG and stress test annually",
                "Immediately report chest pain or breathlessness",
                "Ask about cardiac rehabilitation programs",
            ],
            "🧠 Mental & Social Health": [
                "Join a cardiac patient support group",
                "Talk openly with family about your condition",
                "Seek counseling if feeling anxious or depressed",
                "Stay socially active — avoid isolation",
            ],
        }

        tip_items = list(tips.items())
        col_a, col_b = st.columns(2)

        for i, (category, points) in enumerate(tip_items):
            col = col_a if i % 2 == 0 else col_b
            with col:
                with st.container(border=True):
                    st.markdown(f"**{category}**")
                    for point in points:
                        st.markdown(f"- {point}")

        st.warning(
            "⚠️ These tips are general guidance only. Always follow your cardiologist's "
            "specific instructions — individual treatment plans vary based on severity, age, and other conditions."
        )

st.divider()
st.caption("⚕️ This tool is for educational purposes only and is not a substitute for professional medical advice.")