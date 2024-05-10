from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained Maternal Health Risk model
maternal_model = joblib.load("Maternal_Health_Risk_Model.pkl")

# Load the trained GDM model and scaler
gdm_model, gdm_scaler = joblib.load("GDM_model_and_scaler.pkl")

# Load the trained HyperThyroid model
hyperthyroid_model = joblib.load("AllhypoData_Health_Risk_Model.pkl")

# Load the trained Anemia model and scaler
anemia_model, anemia_scaler = joblib.load("Anemia_model_and_scaler.pkl")

# Define a dictionary to map risk level predictions to labels
risk_levels = {0: 'high risk', 1: 'low risk', 2: 'mid risk'}

# Define a dictionary to map HyperThyroid predictions to labels
hyperthyroid_labels = {0: 'compensated_hypothyroid', 1: 'negative', 2: 'primary_hypothyroid'}


@app.route('/Maternal', methods=['POST'])
def predict_maternal_risk():
    # Get data from the frontend
    data = request.json
    new_data_point = pd.DataFrame({
    "Age": [data["Age"]],
    "SystolicBP": [data["SystolicBP"]],
    "DiastolicBP": [data["DiastolicBP"]],
    "BS": [data["BS"]],
    "BodyTemp": [data["BodyTemp"]],
    "HeartRate": [data["HeartRate"]]
})
    
    
    
    # Create a DataFrame for the new data point with feature names
    new_data_point = pd.DataFrame(data, index=[0])  # Set index explicitly
    
    # Predict the risk level for the new data point
    prediction = maternal_model.predict(new_data_point)
    
    # Decode the predicted risk level
    predicted_risk = risk_levels[prediction[0]]
    
    # Return the predicted risk level
    return jsonify({"predicted_risk": predicted_risk})


@app.route('/GDM', methods=['POST'])
def predict_gdm():
    # Get data from the frontend
    data = request.json
    
    # Create a DataFrame for the new data point with feature names
    new_data_point = pd.DataFrame({
    "BMI": [data["BMI"]],
    "HDL": [data["HDL"]],
    "Sys BP": [data["Sys BP"]],
    "Dia BP": [data["Dia BP"]],
    "Hemoglobin": [data["Hemoglobin"]],
})
    
    # Standardize the new data point using the previously fitted scaler
    new_data_scaled = gdm_scaler.transform(new_data_point)
    
    # Predict the class label for the new data point
    prediction = gdm_model.predict(new_data_scaled)
    
    # Return the predicted result
    return jsonify({"prediction": int(prediction[0])})  # Assuming 0 for Non GDM and 1 for GDM




@app.route('/HyperThyroid', methods=['POST'])
def predict_hyperthyroid():
    # Get data from the frontend
    data = request.json
    
    # Create a DataFrame for the new data point with feature names
    new_data_point = pd.DataFrame({
        "age": [data["age"]],
        "TSH": [data["TSH"]],
        "T3": [data["T3"]],
        "TT4": [data["TT4"]],
        "T4U": [data["T4U"]]
    })
    
    # Predict the target for the new data point
    predicted_target = hyperthyroid_model.predict(new_data_point)
    
    # Decode the predicted target
    predicted_target_str = hyperthyroid_labels[predicted_target[0]]
    
    # Return the predicted target
    return jsonify({"predicted_target": predicted_target_str})


@app.route('/Anemia', methods=['POST'])
def predict_anemia():
    # Get data from the frontend
    data = request.json
    
    
    # Create a DataFrame for the new data point with feature names
    new_data_point = pd.DataFrame({
        "Hemoglobin": [data["Hemoglobin"]],
        "MCH": [data["MCH"]],
        "MCHC": [data["MCHC"]],
        "MCV": [data["MCV"]]
    })
    
    # Standardize the new data point using the previously fitted scaler
    new_data_scaled = anemia_scaler.transform(new_data_point)
    
    # Predict the class label for the new data point
    prediction = anemia_model.predict(new_data_scaled)
    
    # Return the predicted result
    return jsonify({"prediction": int(prediction[0])})


# Route to receive sample data and return top 5 similar data points
@app.route('/find_similar', methods=['POST'])
def find_similar():
    request_data = request.get_json()
    if not request_data:
        return jsonify({"error": "No data provided"}), 400
    
    if 'sample_data' not in request_data:
        return jsonify({"error": "Sample data not provided"}), 400

    sample_data = request_data['sample_data']
    new_data = sample_data[-1]  # Take the last item as the new data

    # Function to calculate similarity between two data points
    def calculate_similarity(data1, data2):
        similarity_score = sum((data1[key] - data2[key])**2 for key in data1.keys() if key != "id")
        return similarity_score

    # Calculate similarity scores for all sample data points except the last one (which is the new data)
    similarity_scores = []
    for data_point in sample_data[:-1]:
        similarity_score = calculate_similarity(new_data, data_point)
        similarity_scores.append((data_point, similarity_score))

    # Sort by similarity score and get top 5
    sorted_similarity_scores = sorted(similarity_scores, key=lambda x: x[1])
    top_5_similar_data = [{ "data": data[0]} for data in sorted_similarity_scores[:5]]

    return jsonify(top_5_similar_data)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000,debug=True)