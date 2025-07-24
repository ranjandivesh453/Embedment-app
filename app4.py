import streamlit as st
import pandas as pd
import pickle

# === Load the trained model ===
with open("RandomForest_HHO_model.pkl", "rb") as file:
    model_loaded = pickle.load(file)

st.set_page_config(page_title="Embedment Depth Prediction", layout="centered")

# === Title ===
st.title("Embedment Depth Prediction using Hybrid RF-HHO Model")
st.markdown("### Prediction of Embedment Depth of Sheet Pile (D)")

# === Input Fields ===
st.subheader("Enter Soil Properties")

# Display labels for UI
display_labels = {
    'gamma1': 'γ₁ (kN/m³)',
    'phi1': 'φ₁ (°)',
    'gamma2': 'γ₂ (kN/m³)',
    'C2': 'C₂ (kPa)'
}

# Mapping for model input
feature_mapping = {
    'gamma1': 'g1',
    'phi1': 'phi1',
    'gamma2': 'g2',
    'C2': 'C2'
}

# Create input fields
input_data = {}
for var, label in display_labels.items():
    input_data[feature_mapping[var]] = st.number_input(label, value=0.0, format="%.2f")

# === Predict Button ===
if st.button("Predict"):
    df_input = pd.DataFrame([input_data])
    prediction = model_loaded.predict(df_input)
    pred_val = round(prediction[0], 3)
    st.success(f"Predicted Depth (D): {pred_val} m")

    # Store session state for export
    st.session_state['latest_prediction'] = (input_data, pred_val)

# === Clear Button ===
if st.button("Clear"):
    for key in feature_mapping.values():
        input_data[key] = 0.0
    st.experimental_rerun()

# === Export Button ===
if 'latest_prediction' in st.session_state:
    input_values, pred_val = st.session_state['latest_prediction']
    df_export = pd.DataFrame([list(input_values.values()) + [pred_val]],
                             columns=list(input_values.keys()) + ['D'])

    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Prediction as CSV",
                       data=csv,
                       file_name='prediction_output.csv',
                       mime='text/csv')
else:
    st.info("Make a prediction to enable CSV export.")
