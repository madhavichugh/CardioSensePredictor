import os
import json
import joblib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

st.set_page_config(
    page_title="CardioSense AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    st.markdown(
        """
<style>
.main{
    background:#f5f7fb;
}
.block-container{
    padding-top:1.5rem;
    padding-bottom:2rem;
}
h1,h2,h3{
    color:#1f2937;
}
div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    border:1px solid #E5E7EB;
    box-shadow:0px 4px 10px rgba(0,0,0,0.05);
}
.stButton>button{
    width:100%;
    height:50px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}
.report-card{
    background:white;
    padding:25px;
    border-radius:15px;
    border:1px solid #dddddd;
    box-shadow:0px 5px 15px rgba(0,0,0,.08);
}
</style>
""",
        unsafe_allow_html=True,
    )

load_css()

MODEL_PATH = "heart_disease_model.pkl"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "cardio_train.csv"

FEATURE_COLUMNS = [
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "age_years",
    "BMI",
    "PulsePressure",
    "age_group",
    "bp_category",
    "risk_score",
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found : {MODEL_PATH}")
        st.stop()
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler not found : {SCALER_PATH}")
        st.stop()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found : {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH, sep=";")
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)
    return df

model, scaler = load_model()
df = load_dataset()

st.sidebar.title("CardioSense AI")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction", "Analytics", "Methodology", "About"],
)

def dashboard_page():
    st.title("📊 CardioSense AI Dashboard")
    st.markdown(
        """
Welcome to **CardioSense AI**.

This dashboard provides an overview of the cardiovascular
disease dataset along with interactive visualizations.
"""
    )
    st.divider()

    total_patients = len(df)
    disease_cases = int(df["cardio"].sum())
    healthy_cases = total_patients - disease_cases
    disease_rate = disease_cases / total_patients * 100
    avg_age = round(df["age"].mean() / 365, 1)
    avg_bmi = round(((df["weight"] / ((df["height"] / 100) ** 2)).mean()), 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Patients", f"{total_patients:,}")
    c2.metric("Disease Cases", disease_cases)
    c3.metric("Healthy", healthy_cases)
    c4.metric("Disease %", f"{disease_rate:.1f}%")
    c5.metric("Average Age", f"{avg_age} yrs")
    st.metric("Average BMI", avg_bmi)

    st.divider()
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.divider()

    st.subheader("Age Distribution")
    age_years = df["age"] / 365
    fig = px.histogram(age_years, nbins=30, title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gender Distribution")
    gender = df["gender"].replace({1: "Female", 2: "Male"})
    fig = px.pie(names=gender, title="Gender")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cholesterol Levels")
    chol = df["cholesterol"].replace({1: "Normal", 2: "Above Normal", 3: "Well Above Normal"})
    fig = px.histogram(chol, title="Cholesterol")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Glucose Levels")
    gluc = df["gluc"].replace({1: "Normal", 2: "Above Normal", 3: "Well Above Normal"})
    fig = px.histogram(gluc, title="Glucose")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("BMI Distribution")
    bmi = df["weight"] / ((df["height"] / 100) ** 2)
    fig = px.histogram(bmi, nbins=35, title="BMI")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Blood Pressure")
    fig = px.scatter(df, x="ap_hi", y="ap_lo", color="cardio", title="Systolic vs Diastolic")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heart Disease Distribution")
    disease = df["cardio"].replace({0: "Healthy", 1: "Disease"})
    fig = px.pie(names=disease, hole=.45, title="Heart Disease")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.success("Dashboard Loaded Successfully")

def prediction_page():
    st.title("Heart Disease Prediction")
    st.markdown(
        """Enter the patient clinical and lifestyle informationto predict the likelihood of cardiovascular disease."""
    )
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (Years)", 18, 100, 45)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        height = st.number_input("Height (cm)", 130, 220, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        ap_hi = st.number_input("Systolic Blood Pressure", 70, 240, 120)
        ap_lo = st.number_input("Diastolic Blood Pressure", 40, 180, 80)

    with col2:
        cholesterol = st.selectbox(
            "Cholesterol",
            [1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x],
        )
        gluc = st.selectbox(
            "Glucose",
            [1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x],
        )
        smoke = st.checkbox("Smoking")
        alco = st.checkbox("Alcohol Consumption")
        active = st.checkbox("Physically Active", value=True)

    st.divider()

    predict = st.button("🔮 Predict Heart Disease", use_container_width=True)
    if not predict:
        return

    if ap_hi <= ap_lo:
        st.error("Systolic BP must be greater than Diastolic BP.")
        return

    bmi = round(weight / ((height / 100) ** 2), 2)
    pulse_pressure = ap_hi - ap_lo

    if age < 40:
        age_group = 0
    elif age < 50:
        age_group = 1
    elif age < 60:
        age_group = 2
    else:
        age_group = 3

    if ap_hi < 120 and ap_lo < 80:
        bp_category = 0
    elif ap_hi < 130 and ap_lo < 80:
        bp_category = 1
    elif ap_hi < 140 or ap_lo < 90:
        bp_category = 2
    else:
        bp_category = 3

    gender_value = 2 if gender == "Male" else 1
    risk_score = cholesterol + gluc + int(smoke) + int(alco) + (1 - int(active))

    patient = pd.DataFrame(
        [[
            gender_value,
            height,
            weight,
            ap_hi,
            ap_lo,
            cholesterol,
            gluc,
            int(smoke),
            int(alco),
            int(active),
            age,
            bmi,
            pulse_pressure,
            age_group,
            bp_category,
            risk_score,
        ]],
        columns=FEATURE_COLUMNS,
    )

    patient_scaled = scaler.transform(patient)
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]
    risk = probability * 100

    st.divider()

    if risk < 30:
        risk_level = "LOW"
        color = "green"
        icon = "🟢"
    elif risk < 70:
        risk_level = "MODERATE"
        color = "orange"
        icon = "🟠"
    else:
        risk_level = "HIGH"
        color = "red"
        icon = "🔴"

    if prediction == 1:
        st.error(f"High Risk of Heart Disease ({risk:.1f}%)")
    else:
        st.success(f"Low Risk of Heart Disease ({100 - risk:.1f}% Confidence)")

    st.subheader("Risk Probability")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk,
            number={"suffix": "%"},
            title={"text": "Heart Disease Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 70], "color": "gold"},
                    {"range": [70, 100], "color": "salmon"},
                ],
            },
        )
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    health_score = max(0, min(100, 100 - risk))
    st.subheader("Overall Health Score")
    st.progress(int(health_score))
    st.metric("Health Score", f"{health_score:.0f}/100")

    st.divider()

    st.subheader("🩺 AI Medical Summary")
    left, right = st.columns(2)

    with left:
        st.metric("Risk Level", risk_level)
        st.metric("Probability", f"{risk:.2f}%")
        st.metric("BMI", bmi)
        st.metric("Pulse Pressure", pulse_pressure)

    with right:
        st.metric("Blood Pressure", f"{ap_hi}/{ap_lo}")
        st.metric("Age", age)
        st.metric("Gender", gender)
        st.metric("Risk Score", risk_score)

    st.divider()

    st.subheader("🏃 Lifestyle Summary")
    l1, l2, l3 = st.columns(3)

    with l1:
        if smoke:
            st.error("🚬 Smoker")
        else:
            st.success("🚭 Non-Smoker")

    with l2:
        if alco:
            st.warning("🍺 Alcohol")
        else:
            st.success("No Alcohol")

    with l3:
        if active:
            st.success("🏃 Physically Active")
        else:
            st.warning("Inactive Lifestyle")

    st.divider()
    st.subheader("💡 Personalized Health Recommendations")

    recommendations = []
    if bmi >= 30:
        recommendations.append("⚖️ Your BMI is high. Aim to lose weight through a healthy diet and regular exercise.")
    elif bmi >= 25:
        recommendations.append("🥗 You are overweight. A modest weight reduction can improve heart health.")
    else:
        recommendations.append("✅ Your BMI is within the healthy range. Keep maintaining it.")

    if ap_hi >= 140 or ap_lo >= 90:
        recommendations.append("🩺 Your blood pressure is elevated. Monitor it regularly and consult a healthcare professional.")
    if cholesterol > 1:
        recommendations.append("🥩 Reduce saturated fats and increase fruits, vegetables, and whole grains.")
    if gluc > 1:
        recommendations.append("🍬 Monitor your blood sugar and reduce sugary foods and drinks.")
    if smoke:
        recommendations.append("🚭 Quitting smoking can significantly reduce cardiovascular risk.")
    if alco:
        recommendations.append("🍺 Reduce alcohol consumption.")
    if not active:
        recommendations.append("🏃 Aim for at least 150 minutes of moderate physical activity each week.")

    for rec in recommendations:
        st.success(rec)

    st.divider()
    st.subheader("🔥 Top Risk Factors")

    risk_factors = []
    if ap_hi >= 140 or ap_lo >= 90:
        risk_factors.append(("High Blood Pressure", 35))
    if bmi >= 30:
        risk_factors.append(("High BMI", 20))
    if cholesterol > 1:
        risk_factors.append(("High Cholesterol", 15))
    if gluc > 1:
        risk_factors.append(("High Glucose", 10))
    if smoke:
        risk_factors.append(("Smoking", 10))
    if alco:
        risk_factors.append(("Alcohol", 5))
    if age >= 60:
        risk_factors.append(("Older Age", 5))

    if len(risk_factors):
        rf = pd.DataFrame(risk_factors, columns=["Risk Factor", "Contribution (%)"])
        st.dataframe(rf, use_container_width=True, hide_index=True)
    else:
        st.success("🎉 No major cardiovascular risk factors detected.")

    st.divider()
    st.subheader("🩺 Clinical Interpretation")

    if prediction == 1:
        st.warning(
            """
The AI model predicts an **elevated likelihood** of cardiovascular disease.

This is **not a medical diagnosis**.

Please consult a qualified healthcare professional for further evaluation.
"""
        )
    else:
        st.success(
            """
The AI model predicts a **low likelihood** of cardiovascular disease.

Maintain healthy habits and attend regular health check-ups.
"""
        )

    st.divider()
    st.subheader("🥗 Diet Suggestions")

    diet = [
        "🥦 Eat more vegetables and fruits.",
        "🌾 Choose whole grains.",
        "🐟 Include lean protein.",
        "🥜 Eat healthy fats.",
        "🧂 Reduce salt intake.",
        "🍩 Limit processed foods.",
        "💧 Drink adequate water.",
    ]
    for item in diet:
        st.write(item)

    st.divider()
    st.subheader("🏃 Exercise Suggestions")

    exercise = [
        "🚶 Walk at least 30 minutes daily.",
        "🚴 Cycle or jog 3–5 times per week.",
        "💪 Perform strength training twice a week.",
        "🧘 Practice yoga or stretching.",
        "😴 Sleep 7–8 hours every night.",
    ]
    for item in exercise:
        st.write(item)

    st.divider()
    st.subheader("📋 Final Assessment")

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Level", risk_level)
    c2.metric("Risk Probability", f"{risk:.1f}%")
    c3.metric("Health Score", f"{health_score:.0f}/100")
    st.success("✅ Assessment completed successfully.")

def analytics_page():
    st.title("📊 Data Analytics")
    st.markdown(
        """
Explore the cardiovascular disease dataset using
interactive charts and summary statistics.
"""
    )
    st.divider()

    st.subheader("📋 Dataset Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.divider()
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.divider()

    st.subheader("❤️ Disease Distribution")
    disease = df["cardio"].replace({0: "Healthy", 1: "Heart Disease"})
    fig = px.pie(names=disease, hole=.45, title="Disease Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("👤 Age Distribution")
    age = df["age"] / 365
    fig = px.histogram(age, nbins=30, title="Age")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚖ BMI Distribution")
    bmi = df["weight"] / ((df["height"] / 100) ** 2)
    fig = px.histogram(bmi, nbins=35, title="BMI")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🩸 Blood Pressure")
    fig = px.scatter(df, x="ap_hi", y="ap_lo", color="cardio", title="Blood Pressure")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🥩 Cholesterol")
    chol = df["cholesterol"].replace({1: "Normal", 2: "Above Normal", 3: "Well Above Normal"})
    fig = px.histogram(chol, title="Cholesterol")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🍬 Glucose")
    gluc = df["gluc"].replace({1: "Normal", 2: "Above Normal", 3: "Well Above Normal"})
    fig = px.histogram(gluc, title="Glucose")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("👥 Gender Distribution")
    gender = df["gender"].replace({1: "Female", 2: "Male"})
    fig = px.pie(names=gender, title="Gender")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Correlation Matrix")
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📄 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.divider()
    st.subheader("⬇ Download Dataset")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "cardio_dataset.csv", "text/csv")
    st.success("Analytics Loaded Successfully")

def methodology_page():
    st.title("⚙️ Methodology")
    st.markdown(
        """
This application predicts cardiovascular disease using a supervised
Machine Learning model trained on the Cardiovascular Disease Dataset.
"""
    )
    st.divider()

    st.subheader("📌 Workflow")
    st.markdown(
        """
1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Feature Scaling
5. Model Training
6. Cross Validation
7. Hyperparameter Tuning
8. Model Evaluation
9. Deployment using Streamlit
"""
    )
    st.divider()

    st.subheader("🧠 Feature Engineering")
    feature_df = pd.DataFrame(
        {
            "Feature": ["Age", "BMI", "Pulse Pressure", "Blood Pressure Category", "Risk Score"],
            "Description": ["Age in years", "Weight / Height²", "Systolic - Diastolic", "Blood pressure classification", "Lifestyle risk index"],
        }
    )
    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    st.divider()

    st.subheader("🤖 Machine Learning")
    st.markdown(
        """
Models Compared

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost (Optional)

Best model selected using

- 5 Fold Cross Validation
- Hyperparameter Tuning
"""
    )
    st.success("Methodology Loaded Successfully")

def about_page():
    st.title("ℹ About CardioSense AI")
    st.markdown(
        """
## 🫀 CardioSense AI

CardioSense AI is a Machine Learning powered web application
designed to estimate the likelihood of cardiovascular disease
using clinical and lifestyle information.

This project demonstrates end-to-end data science,
machine learning, and interactive visualization.
"""
    )
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📦 Technologies")
        st.markdown(
            """
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Joblib
"""
        )

    with c2:
        st.subheader("📊 Dataset")
        st.markdown(
            """
- Cardiovascular Disease Dataset
- ~70,000 Records
- 11 Original Features
- Engineered Clinical Features
"""
        )

    st.divider()
    st.subheader("⚠ Disclaimer")
    st.warning(
        """
This application is intended for educational and research purposes.

Predictions generated by the AI model should not be considered
medical advice or a clinical diagnosis.
Always consult a qualified healthcare professional.
"""
    )
    st.success("About Page Loaded Successfully")

def main():
    if page == "Dashboard":
        dashboard_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "Methodology":
        methodology_page()
    elif page == "About":
        about_page()

    st.sidebar.markdown("---")
    st.sidebar.info(
        "🫀 CardioSense AI\n\n"
        "Machine Learning Based Heart Disease Prediction System"
    )

if __name__ == "__main__":
    main()