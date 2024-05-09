import pandas as pd
import joblib

# Load the trained model
model = joblib.load("AllhypoData_Health_Risk_Model.pkl")

# Define the new data point
new_data_point = pd.DataFrame({
    "age": [27],
    "TSH": [15],
    "T3": [1.6],
    "TT4": [82],
    "T4U": [.82]

})

# Predict the target for the new data point
predicted_target = model.predict(new_data_point)

# Define the encoding used for the target variable
target_encoding = {
    0: 'compensated_hypothyroid',
    1: 'negative',
    2:'primary_hypothyroid',
}

# Inverse transform the predicted target
predicted_target_str = target_encoding[predicted_target[0]]

print("Predicted target:", predicted_target_str)
