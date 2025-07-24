import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("RandomForest_HHO_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Embedment Depth Prediction using Hybrid RF-HHO Model")

# Inputs
gamma1 = st.number_input("γ₁ (kN/m³)", value=18.0)
phi1 = st.number_input("φ₁ (°)", value=30.0)
gamma2 = st.number_input("γ₂ (kN/m³)", value=19.0)
C2 = st.number_input("C₂ (kPa)", value=10.0)

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([{
        'g1': gamma1,
        'phi1': phi1,
        'g2': gamma2,
        'C2': C2
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Embedment Depth (D): {round(prediction, 3)}")
