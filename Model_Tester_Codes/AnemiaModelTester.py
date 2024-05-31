import pandas as pd
import joblib

# Load the model and scaler from the file
model, scaler = joblib.load("Anemia_model_and_scaler.pkl")

# Create a DataFrame for the new data point with feature names
new_data_point = pd.DataFrame({
    "Hemoglobin": [14.8],
    "MCH": [23.4],
    "MCHC": [29.2],
    "MCV": [74.7]
})

# Standardize the new data point using the previously fitted scaler
new_data_scaled = scaler.transform(new_data_point)

# Predicting the class label for the new data point
prediction = model.predict(new_data_scaled)
print("Prediction for new data:", prediction)
