import pandas as pd 
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/processed/alzheimers_prediction_dataset.csv')

x = data.drop('Alzheimer’s Diagnosis', axis=1)
y = data['Alzheimer’s Diagnosis']

if y.dtype != 'object':  
    y = (y > y.median()).astype(int)


X = pd.get_dummies(x, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")


with open('models/alzheimerModel.pkl', 'wb') as f:
    pickle.dump(model, f)



