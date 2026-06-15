import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
st.title("Bank Churn Prediction")
st.subheader("Top Churn Drivers")
feature_df = pd.read_csv("feature_importance.csv")
st.dataframe(feature_df.head(10))
fig, ax = plt.subplots(figsize=(8,5))
top_features = feature_df.head(10)
ax.barh(
    top_features["Feature"],
    top_features["Importance"]
)
ax.set_title("Top 10 Churn Drivers")
ax.set_xlabel("Importance")

st.pyplot(fig)
year = st.number_input("Year", min_value=2020, max_value=2030, value=2025)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.number_input("Num Of Products", min_value=1, max_value=4, value=1)

has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])

salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

germany = st.selectbox("Germany", [0, 1])
spain = st.selectbox("Spain", [0, 1])
male = st.selectbox("Male", [0, 1])

if st.button("Predict"):

    # Feature Engineering
    balance_salary_ratio = balance / (salary + 1)
    product_density = num_products / (age + 1)
    engagement_product = is_active * num_products
    age_tenure = age * tenure

    data = np.array([[
        year,
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_card,
        is_active,
        salary,
        balance_salary_ratio,
        product_density,
        engagement_product,
        age_tenure,
        germany,
        spain,
        male
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    probability = model.predict_proba(data_scaled)[0][1]

st.write("Churn Probability:", round(probability * 100, 2), "%")
st.progress(int(probability * 100))
fig, ax = plt.subplots()
ax.bar(
    ["Stay", "Leave"],
    [1 - probability, probability]
)
ax.set_title("Churn Probability Distribution")
ax.set_ylabel("Probability")
st.pyplot(fig)
# Risk Level
if probability < 0.30:
    st.success("Risk Level: LOW")
elif probability < 0.70:
    st.warning("Risk Level: MEDIUM")
else:
    st.error("Risk Level: HIGH")

# Final Prediction
if prediction[0] == 1:
    st.error("Customer Will Leave")
else:
    st.success("Customer Will Stay")