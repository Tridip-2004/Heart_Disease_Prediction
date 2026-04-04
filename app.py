from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model     = joblib.load("best_model.pkl")
scaler    = joblib.load("scaler.pkl")
rfe       = joblib.load("rfe_selector.pkl")
threshold = joblib.load("threshold.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction  = None
    probability = None
    tips        = None

    if request.method == "POST":
        # ── 1. Collect raw inputs ──────────────────────────────
        age      = float(request.form["age"])
        sex      = float(request.form["sex"])
        cp       = float(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol     = float(request.form["chol"])
        fbs      = float(request.form["fbs"])
        restecg  = float(request.form["restecg"])
        thalach  = float(request.form["thalach"])
        exang    = float(request.form["exang"])
        oldpeak  = float(request.form["oldpeak"])
        slope    = float(request.form["slope"])
        ca       = float(request.form["ca"])
        thal     = float(request.form["thal"])

        # ── 2. Build DataFrame with same column names as training ──
        new = pd.DataFrame([{
            'Age'                  : age,
            'Sex'                  : sex,
            'ChestPainType'        : cp,
            'RestingBloodPressure' : trestbps,
            'Cholesterol'          : chol,
            'FastingBloodSugar'    : fbs,
            'RestingECG'           : restecg,
            'MaxHeartRate'         : thalach,
            'ExerciseInducedAngina': exang,
            'Oldpeak'              : oldpeak,
            'SlopeOfST_Segment'    : slope,
            'MajorVesselsColored'  : ca,
            'Thalassemia'          : thal
        }])

        # ── 3. Engineered features (same as notebook) ─────────
        new['Age_MaxHR_Ratio']  = new['Age'] / (new['MaxHeartRate'] + 1)
        new['BP_Chol_Ratio']    = new['RestingBloodPressure'] / (new['Cholesterol'] + 1)
        new['Age_Oldpeak']      = new['Age'] * new['Oldpeak']
        new['MaxHR_Oldpeak']    = new['MaxHeartRate'] / (new['Oldpeak'] + 1)
        new['ChestPain_Angina'] = new['ChestPainType'] * new['ExerciseInducedAngina']
        new['Slope_Oldpeak']    = new['SlopeOfST_Segment'] * new['Oldpeak']

        # Age_Bin and MaxHR_Bin — replicate notebook binning logic
        new['Age_Bin']   = pd.cut([age],    bins=[0, 40, 55, 70, 120], labels=[0, 1, 2, 3]).astype(float)[0]
        new['MaxHR_Bin'] = pd.cut([thalach], bins=[0, 100, 140, 170, 300], labels=[0, 1, 2, 3]).astype(float)[0]

        # ── 4. Scale → RFE → Predict ──────────────────────────
        X_sc  = scaler.transform(new)
        X_rfe = rfe.transform(X_sc)

        prob        = float(model.predict_proba(X_rfe)[0][1])
        prediction  = int(prob >= threshold)
        probability = int(prob * 100)

        # ── 5. Tips ───────────────────────────────────────────
        if prediction == 1:
            tips = [
                "Consult a cardiologist immediately",
                "Reduce salt and saturated fat intake",
                "Avoid smoking and alcohol",
                "Exercise only as advised by doctor",
                "Monitor BP and cholesterol regularly",
                "Practice stress management (yoga, meditation)"
            ]
        else:
            tips = [
                "Maintain a balanced diet",
                "Exercise at least 30 minutes daily",
                "Avoid smoking and excessive alcohol",
                "Maintain healthy weight",
                "Control cholesterol and BP",
                "Get regular health checkups"
            ]

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        tips=tips
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)