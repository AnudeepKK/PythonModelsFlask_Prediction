import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Read the CSV file
data = pd.read_csv("Datasets/Gestational Diabetic Dat Set.csv")

# Remove rows with any missing values
cleaned_data = data.dropna()

# Split features and target variable
X = cleaned_data.drop(columns=["Class Label(GDM /Non GDM)", "Case Number"])  # Remove "Case Number" column
y = cleaned_data["Class Label(GDM /Non GDM)"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Predicting a new data point
# Creating a DataFrame for the new data point with feature names
new_data_point = pd.DataFrame({
    "Age": [39],
    "No of Pregnancy": [3],
    "Gestation in previous Pregnancy": [2],
    "BMI": [32.1],
    "HDL": [31],
    "Family History": [1],
    "PCOS": [0],
    "Sys BP": [139],
    "Dia BP": [80],
    "Hemoglobin": [12.8],
    "Sedentary Lifestyle": [1],
    "Prediabetes": [1],
})

# Standardize the new data point using the previously fitted scaler
new_data_scaled = scaler.transform(new_data_point)

# Predicting the class label for the new data point
prediction = model.predict(new_data_scaled)
print("Prediction for new data:", prediction)
