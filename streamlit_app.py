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

# User input
st.write("Enter 8 input features:")
features = []
for i in range(8):
    features.append(st.number_input(f"Feature {i+1}", step=0.1))

if st.button("Predict"):
    input_array = np.array(features).reshape(1, -1)

    # Assume the scaler was fitted on full dataset; simulate loading it
    df = pd.read_excel("ENB2012_data.xlsx")
    scaler = StandardScaler()
    scaler.fit(df.iloc[:, 0:8].values)

    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    st.success(f"Predicted Heating Load: {prediction:.2f}")
