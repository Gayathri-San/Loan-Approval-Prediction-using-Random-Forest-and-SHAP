import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------
# Load Model
# -----------------------
model = joblib.load(open("rf.pkl", "rb"))

st.set_page_config(page_title="Loan Approval Prediction", layout="wide")
st.title("🏦 Loan Approval with Risk Explanation")

# -----------------------
# Mappings
# -----------------------
occupation_status_map = {'Employed': 1, 'Self-Employed': 2, 'Student': 3}
product_type_map = {'Credit Card': 1, 'Personal Loan': 2, 'Line of Credit': 3}
loan_intent_map = {
    'Personal': 1, 'Education': 2, 'Medical': 3,
    'Business': 4, 'Home Improvement': 5, 'Debt Consolidation': 6
}

# -----------------------
# Layout
# -----------------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", key="age")
    occupation_status = st.selectbox("Occupation Status", list(occupation_status_map.keys()), key="occupation_status")
    years_employed = st.number_input("Years Employed", key="years_employed")
    annual_income = st.number_input("Annual Income", key="annual_income")
    credit_score = st.number_input("Credit Score", key="credit_score")
    credit_history_years = st.number_input("Credit History Years", key="credit_history_years")

with col2:
    savings_assets = st.number_input("Savings / Assets", key="savings_assets")
    current_debt = st.number_input("Current Debt", key="current_debt")
    defaults_on_file = st.number_input("Defaults On File", key="defaults_on_file")
    delinquencies_last_2yrs = st.number_input("Delinquencies Last 2 Years", key="delinquencies_last_2yrs")
    derogatory_marks = st.number_input("Derogatory Marks", key="derogatory_marks")
    product_type = st.selectbox("Product Type", list(product_type_map.keys()), key="product_type")

with col3:
    loan_intent = st.selectbox("Loan Intent", list(loan_intent_map.keys()), key="loan_intent")
    loan_amount = st.number_input("Loan Amount", key="loan_amount")
    interest_rate = st.number_input("Interest Rate", key="interest_rate")
    debt_to_income_ratio = st.number_input("Debt to Income Ratio", key="debt_to_income_ratio")
    loan_to_income_ratio = st.number_input("Loan to Income Ratio", key="loan_to_income_ratio")
    payment_to_income_ratio = st.number_input("Payment to Income Ratio", key="payment_to_income_ratio")

# -----------------------
# DataFrame
# -----------------------
input_data = pd.DataFrame({
    "age": [age],
    "occupation_status": [occupation_status_map[occupation_status]],
    "years_employed": [years_employed],
    "annual_income": [annual_income],
    "credit_score": [credit_score],
    "credit_history_years": [credit_history_years],
    "savings_assets": [savings_assets],
    "current_debt": [current_debt],
    "defaults_on_file": [defaults_on_file],
    "delinquencies_last_2yrs": [delinquencies_last_2yrs],
    "derogatory_marks": [derogatory_marks],
    "product_type": [product_type_map[product_type]],
    "loan_intent": [loan_intent_map[loan_intent]],
    "loan_amount": [loan_amount],
    "interest_rate": [interest_rate],
    "debt_to_income_ratio": [debt_to_income_ratio],
    "loan_to_income_ratio": [loan_to_income_ratio],
    "payment_to_income_ratio": [payment_to_income_ratio]
})

# -----------------------
# Prediction
# -----------------------
if st.button("Predict"):

    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    # -----------------------
    # SHAP
    # -----------------------
    st.subheader("🔍 Why this decision?")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    shap_single = shap_values[0, :, prediction[0]]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_single, max_display=len(input_data.columns), show=False)
    st.pyplot(fig)