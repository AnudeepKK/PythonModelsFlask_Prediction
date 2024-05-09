import pandas as pd
import joblib

# Load the trained model
model = joblib.load("Maternal_Health_Risk_Model.pkl")

# Create a DataFrame for the new data point with feature names
new_data_point = pd.DataFrame({
    "Age": [22],
    "SystolicBP": [120],
    "DiastolicBP": [90],
    "BS": [7.1],
    "BodyTemp": [98],
    "HeartRate": [82]
})

# Predicting the risk level for the new data point
prediction = model.predict(new_data_point)

# Decode the predicted risk level
risk_levels = {0: 'high risk', 1: 'low risk', 2: 'mid risk'}  # Assuming you have encoded risk levels as 0, 1, and 2
predicted_risk = risk_levels[prediction[0]]
print("Predicted Risk Level:", predicted_risk)
