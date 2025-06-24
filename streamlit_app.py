import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Define the same model architecture
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load model
model = RegressionModel()
model.load_state_dict(torch.load("regression_model.pth", map_location=torch.device('cpu')))
model.eval()

# App title
st.title("Heating Load Prediction App")
st.markdown("Enter the building characteristics below to estimate the **heating load**.")

# Sidebar info
st.sidebar.header("ℹ️ Model Information")
st.sidebar.markdown("- **Model:** PyTorch Feedforward Neural Network")
st.sidebar.markdown("- **Trained on:** UCI ENB Dataset")
st.sidebar.markdown("- **Target:** Heating Load (Y1)")
st.sidebar.markdown("- **Input features:** 8 building parameters")
st.sidebar.markdown("- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/energy+efficiency)")


#input
st.write("### Building Characteristics")
features = []

feature_inputs = {
    "Relative Compactness": {"min_value": 0.5, "max_value": 1.0, "step": 0.01, "note": "Typically ranges from 0.5 to 1.0"},
    "Surface Area": {"min_value": 500.0, "max_value": 900.0, "step": 1.0, "note": "Total external surface in m²"},
    "Wall Area": {"min_value": 200.0, "max_value": 400.0, "step": 1.0, "note": "Total wall area in m²"},
    "Roof Area": {"min_value": 100.0, "max_value": 300.0, "step": 1.0, "note": "Roof area in m²"},
    "Overall Height": {"min_value": 3.5, "max_value": 7.0, "step": 0.5, "note": "Building height — 3.5 or 7.0"},
    "Orientation": {"min_value": 2, "max_value": 5, "step": 1, "note": "Integer from 2 to 5"},
    "Glazing Area": {"min_value": 0.0, "max_value": 0.4, "step": 0.01, "note": "Ratio from 0.0 to 0.4"},
    "Glazing Area Distribution": {"min_value": 0, "max_value": 5, "step": 1, "note": "Integer from 0 to 5"}
}

for label, params in feature_inputs.items():
    st.caption(params["note"])
    features.append(st.number_input(label, min_value=params["min_value"], max_value=params["max_value"], step=params["step"]))


# Prediction 
if st.button("Predict"):
    input_array = np.array(features).reshape(1, -1)

    # Load and fit the scaler on the full dataset
    df = pd.read_excel("ENB2012_data.xlsx")
    scaler = StandardScaler()
    scaler.fit(df.iloc[:, 0:8].values)

    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    st.success(f" Predicted Heating Load: **{prediction:.2f}**")
