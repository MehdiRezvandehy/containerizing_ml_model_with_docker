import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -----------------------------
# 1. Load Model and Preprocessors
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("fraud_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    return model, scaler, le

model, scaler, le = load_artifacts()

# -----------------------------
# 2. Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Realistic Fraud Detection App", layout="centered")

st.title("💳 Transaction Fraud Detection")
st.write("Model trained on the **PaySim synthetic transaction dataset** — features mimic real financial data.")

# Input fields
st.subheader("Enter Transaction Details")

transaction_type = st.selectbox(
    "Transaction Type",
    le.classes_.tolist()
)
amount = st.number_input("💰 Transaction Amount ($)", 0.0, 100000.0, 2500.0, step=100.0)
oldbalanceOrg = st.number_input("🏦 Sender’s Old Balance", 0.0, 1000000.0, 5000.0, step=100.0)
newbalanceOrig = st.number_input("🏦 Sender’s New Balance", 0.0, 1000000.0, 2500.0, step=100.0)
oldbalanceDest = st.number_input("💼 Receiver’s Old Balance", 0.0, 1000000.0, 10000.0, step=100.0)
newbalanceDest = st.number_input("💼 Receiver’s New Balance", 0.0, 1000000.0, 12500.0, step=100.0)

# Convert to model input
type_encoded = le.transform([transaction_type])[0]
input_df = pd.DataFrame([{
    'type': type_encoded,
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest
}])

# -----------------------------
# 3. Prediction Trigger Button
# -----------------------------
if st.button("🔍 Predict Fraud"):
    # Scale input
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    st.write("---")
    st.subheader("🔍 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability of Fraud: {prob:.2f})")

else:
    st.info("🧭 Fill out the form above and click **'🔍 Predict Fraud'** to see results.")

st.caption("This demo uses the PaySim synthetic dataset (simulating mobile money transactions).")
