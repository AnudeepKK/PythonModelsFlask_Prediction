import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Read the CSV file
data = pd.read_csv("Gestational Diabetic Dat Set.csv")

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

# Save the trained model and scaler to a file
joblib.dump((model, scaler), "GDM_model_and_scaler.pkl")

# Now you can load the model and scaler from the file and use them for prediction
