#  Heating Load Prediction App

A microservice web application that predicts the **heating load (Y1)** of a building using a neural network regression model trained on the **ENB2012 dataset**. Built with **PyTorch** and deployed using **Streamlit Cloud**, the app allows users to interactively input building parameters and receive real-time energy performance predictions.

![Streamlit App Screenshot](<img width="391" alt="image" src="https://github.com/user-attachments/assets/873bd83d-c1c4-46c1-a2fe-6fd24d7d97ec" />
)

---

##  Live Demo

 [Click here to try the app](https://eai6010module5-rk6wljpa3yzrfwwf4espwz.streamlit.app/)

---

##  Project Overview

The app predicts heating energy demand (in **kWh/m²**) based on 8 key architectural design inputs:

- Relative Compactness
- Surface Area
- Wall Area
- Roof Area
- Overall Height
- Orientation
- Glazing Area
- Glazing Area Distribution

### Example Input

| Feature                    | Value      |
|---------------------------|------------|
| Relative Compactness      | 0.80       |
| Surface Area              | 750.00     |
| Wall Area                 | 300.00     |
| Roof Area                 | 200.00     |
| Overall Height            | 3.50       |
| Orientation               | 2          |
| Glazing Area              | 0.10       |
| Glazing Area Distribution | 3          |

### Example Output
>  **Predicted Heating Load**: `8.85 kWh/m²`

---

## 🧠 Model Details

- **Framework**: PyTorch  
- **Model Architecture**: Feedforward Neural Network  
- **Preprocessing**: Scikit-learn `StandardScaler`  
- **Training Data**: [ENB2012 Dataset](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)

---

## 📦 Files Included

- `streamlit_app.py` – Streamlit frontend and model inference logic  
- `regression_model.pth` – Trained PyTorch model weights  
- `ENB2012_data.xlsx` – Dataset for scaling input features  
- `requirements.txt` – Python dependencies for Streamlit Cloud

---

## 🛠️ How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/aparey/EAI_6010_Module5.git
   
