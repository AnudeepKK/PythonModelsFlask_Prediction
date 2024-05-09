import pandas as pd
import joblib

# Load the model and scaler from the file
model, scaler = joblib.load("GDM_model_and_scaler.pkl")

# Create a DataFrame for the new data point with feature names
new_data_point = pd.DataFrame({
    "BMI": [32.1],
    "HDL": [31],
    "Sys BP": [139],
    "Dia BP": [80],
    "Hemoglobin": [12.8],
    "Prediabetes": [1],
})

# Standardize the new data point using the previously fitted scaler
new_data_scaled = scaler.transform(new_data_point)

# Predicting the class label for the new data point
prediction = model.predict(new_data_scaled)
print("Prediction for new data:", prediction)
