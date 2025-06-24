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

# feature labels
feature_inputs = {
    "Relative Compactness": {"min": 0.5, "max": 1.0, "step": 0.01, "help": "Ratio of volume to surface area"},
    "Surface Area": {"min": 500.0, "max": 900.0, "step": 1.0, "help": "Total external surface (m²)"},
    "Wall Area": {"min": 200.0, "max": 400.0, "step": 1.0, "help": "Total wall area (m²)"},
    "Roof Area": {"min": 100.0, "max": 300.0, "step": 1.0, "help": "Total roof area (m²)"},
    "Overall Height": {"min": 3.5, "max": 7.0, "step": 0.5, "help": "Height of the building (3.5 or 7.0)"},
    "Orientation": {"min": 2, "max": 5, "step": 1, "help": "Integer (2–5) indicating orientation"},
    "Glazing Area": {"min": 0.0, "max": 0.4, "step": 0.01, "help": "Ratio of glazing to wall area"},
    "Glazing Area Distribution": {"min": 0, "max": 5, "step": 1, "help": "Integer (0–5) for glazing layout"}
}

# User input
st.write("### Building Characteristics")
features = []
for label, config in feature_inputs.items():
    features.append(st.number_input(label, **config))

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
