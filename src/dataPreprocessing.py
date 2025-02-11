import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('data/raw/alzheimers_prediction_dataset.csv')
print(data.head())

data = data.dropna()

label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data = pd.DataFrame(data_scaled, columns=data.columns)

data.to_csv('data/processed/alzheimers_prediction_dataset.csv', index=False)

print("Preprocesamiento completado. Datos guardados en 'data/processed/processed_data.csv'")


