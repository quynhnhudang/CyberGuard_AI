import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess dataset
data = pd.read_csv('CICIDS2017.csv')
data = data.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], axis=1)
data.fillna(0, inplace=True)
scaler = StandardScaler()
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
data.to_csv('preprocessed_data.csv', index=False)
