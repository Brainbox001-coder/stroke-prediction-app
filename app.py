import streamlit as st
import pandas as pd
import joblib

# Load the trained machine learning pipeline
model = joblib.load("stroke_model.pkl")

st.set_page_config(
    page_title="Group 13 | Stroke Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Stroke Prevention using Machine Learning Techniques")
st.subheader("Group 13 — Stroke Risk Prediction")

st.write(
    "Enter the demographic and health-related information "
    "below to obtain a prediction from our trained "
    "Logistic Regression model."
)
st.warning(
   "This application is an educational capstone project. "
    "Its predictions are not a medical diagnosis and must "
    "not be used to make clinical decisions."
)

with st.form("stroke_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender", ["Male", "Female", "Other"]
        )
        age = st.number_input(
            "Age", min_value=0.0, max_value=120.0, value=45.0
        )
        hypertension = st.selectbox(
            "Hypertension", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        heart_disease = st.selectbox(
            "Heart Disease", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        ever_married = st.selectbox(
            "Ever Married", ["Yes", "No"]
        )

    with col2:
        work_type = st.selectbox(
            "Work Type",
            ["Private", "Self-employed", "Govt_job",
             "children", "Never_worked"]
        )
        Residence_type = st.selectbox(
            "Residence Type", ["Urban", "Rural"]
        )
        avg_glucose_level = st.number_input(
            "Average Glucose Level",
            min_value=0.0, value=100.0
        )
        bmi = st.number_input(
            "BMI", min_value=0.0, value=25.0
        )
        smoking_status = st.selectbox(
            "Smoking Status",
            ["formerly smoked", "never smoked",
             "smokes", "Unknown"]
        )

    submitted = st.form_submit_button("Predict Stroke Classification")

if submitted:
    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }])

    try:
        prediction = int(model.predict(input_data)[0])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Model classification: Stroke")
        else:
            st.success("Model classification: No Stroke")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            st.write(
                f"Model-estimated stroke-class probability: "
                f"{probabilities[1] * 100:.2f}%"
            )

        st.caption(
            "This result is a model output, not a diagnosis or "
            "a validated estimate of an individual's medical risk."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the saved pipeline matches the input "
            "feature names and categories."
        )

        st.divider()

st.caption(
    "Group 13 | TechCrush AI/ML Capstone Project | October 2026"
)
