import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances

# Function to preprocess the data
def preprocess_data(df):
    # Drop the 'RiskLevel' column temporarily for imputation
    df_temp = df.drop(columns=['RiskLevel'])

    # Impute missing values with mean for numeric columns
    imputer = SimpleImputer(strategy='mean')
    df_temp_imputed = pd.DataFrame(imputer.fit_transform(df_temp), columns=df_temp.columns)

    # Concatenate 'RiskLevel' column back to the imputed DataFrame
    df_imputed = pd.concat([df_temp_imputed, df['RiskLevel']], axis=1)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_imputed.drop(columns=['RiskLevel']))

    return X_scaled, scaler, imputer

# Read data from the CSV file
data = pd.read_csv('Datasets/Maternal Health Risk Data Set.csv')

# Preprocess the data
X_scaled, scaler, imputer = preprocess_data(data)

# Prepare the new data provided by the user
new_data = pd.DataFrame({
    'Age': [26],
    'SystolicBP': [110],
    'DiastolicBP': [80],
    'BS': [7],
    'BodyTemp': [98],
    'HeartRate': [76]
})

# Preprocess the new data using the same scaler and imputer
new_data_imputed = pd.DataFrame(imputer.transform(new_data), columns=new_data.columns)
new_data_scaled = scaler.transform(new_data_imputed)

# Calculate distances between the new data and existing data points
distances = pairwise_distances(X_scaled, new_data_scaled, metric='euclidean').ravel()

# Combine distances with original data
data['Distance'] = distances

# Sort data by distance
data = data.sort_values(by='Distance')

# Get top five similar individuals
top_five_similar = data.head(5)

# Display top five similar individuals
# Display top five similar individuals in tabular format with index and all columns
print("Top 5 individuals with similar risks:")
print(top_five_similar.to_string(index=True))

