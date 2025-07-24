{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": none,
   "id": "9010443d-e70f-450f-8661-a55addbf03b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pickle\n",
    "\n",
    "# === Load the trained model ===\n",
    "@st.cache_resource\n",
    "def load_model():\n",
    "    with open(\"RandomForest_HHO_model.pkl\", \"rb\") as file:\n",
    "        return pickle.load(file)\n",
    "\n",
    "model_loaded = load_model()\n",
    "\n",
    "# === Streamlit UI ===\n",
    "st.set_page_config(page_title=\"Embedment Depth Prediction\", layout=\"centered\")\n",
    "st.title(\"Embedment Depth Prediction of Sheet Pile (D) using RF-HHO Model\")\n",
    "st.markdown(\"Enter the input parameters below to predict the embedment depth.\")\n",
    "\n",
    "# Input Fields\n",
    "gamma1 = st.number_input(\"γ₁ (kN/m³)\", min_value=0.0, value=18.0)\n",
    "phi1 = st.number_input(\"φ₁ (°)\", min_value=0.0, value=30.0)\n",
    "gamma2 = st.number_input(\"γ₂ (kN/m³)\", min_value=0.0, value=19.0)\n",
    "C2 = st.number_input(\"C₂ (kPa)\", min_value=0.0, value=10.0)\n",
    "\n",
    "# Prediction\n",
    "if st.button(\"Predict\"):\n",
    "    input_df = pd.DataFrame([{\n",
    "        'g1': gamma1,\n",
    "        'phi1': phi1,\n",
    "        'g2': gamma2,\n",
    "        'C2': C2\n",
    "    }])\n",
    "    prediction = model_loaded.predict(input_df)\n",
    "    st.success(f\"Predicted Embedment Depth (D): {round(prediction[0], 3)} m\")\n",
    "\n",
    "# Export Option\n",
    "if st.button(\"Export Prediction to CSV\"):\n",
    "    input_data = pd.DataFrame([{\n",
    "        'g1': gamma1,\n",
    "        'phi1': phi1,\n",
    "        'g2': gamma2,\n",
    "        'C2': C2,\n",
    "        'D': round(prediction[0], 3) if 'prediction' in locals() else None\n",
    "    }])\n",
    "    csv = input_data.to_csv(index=False).encode(\"utf-8\")\n",
    "    st.download_button(\n",
    "        label=\"Download CSV\",\n",
    "        data=csv,\n",
    "        file_name=\"embedment_depth_prediction.csv\",\n",
    "        mime=\"text/csv\"\n",
    "    )\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
