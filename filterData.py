import pandas as pd

# Read the CSV file
data = pd.read_csv("AllhypoData.csv")

# Remove rows where sex is 'M'
cleaned_data = data[data['sex'] != 'M']

# Remove rows with missing values represented as '?'
cleaned_data = cleaned_data.replace('?', pd.NA).dropna()

# Drop duplicates to ensure only unique, complete rows are retained
cleaned_data = cleaned_data.drop_duplicates()

# Write cleaned data to a new CSV file
cleaned_data.to_csv("cleaned_AllhypoData.csv", index=False)
