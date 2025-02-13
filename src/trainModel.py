import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Cargar los datos
data = pd.read_csv('data/processed/alzheimersPredictionDataset.csv')

# Separar características y etiquetas
xData = data.drop('AlzheimersDiagnosis', axis=1)
yData = data['AlzheimersDiagnosis']

# Convertir a valores binarios si es necesario
if yData.dtype != 'object':  
    yData = (yData > yData.median()).astype(int)

# Codificar variables categóricas
xDataEncoded = pd.get_dummies(xData, drop_first=True)
xTrain, xTest, yTrain, yTest = train_test_split(xDataEncoded, yData, test_size=0.2, random_state=42)

# Definir hiperparámetros a probar
paramGrid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Crear el modelo base
randomForest = RandomForestClassifier(random_state=42)

# RandomizedSearchCV para encontrar la mejor combinación de hiperparámetros
rfSearch = RandomizedSearchCV(randomForest, paramGrid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
rfSearch.fit(xTrain, yTrain)

# Mejor modelo encontrado
bestModel = rfSearch.best_estimator_
print("Mejores hiperparámetros:", rfSearch.best_params_)

# Evaluación del modelo optimizado
yPredOpt = bestModel.predict(xTest)
accuracyOpt = accuracy_score(yTest, yPredOpt)
print(f"Nueva precisión: {accuracyOpt:.2f}")

# Matriz de confusión
confMatrix = confusion_matrix(yTest, yPredOpt)
plt.figure(figsize=(6, 4))
sns.heatmap(confMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Alzheimer', 'Alzheimer'], yticklabels=['No Alzheimer', 'Alzheimer'])
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# Reporte de clasificación
print("Reporte de Clasificación:")
print(classification_report(yTest, yPredOpt))

# Curva ROC y AUC
if len(yData.unique()) == 2:  # Solo para clasificación binaria
    yProb = bestModel.predict_proba(xTest)[:, 1]  # Probabilidad de la clase positiva
    fpr, tpr, _ = roc_curve(yTest, yProb)
    rocAuc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {rocAuc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()

# Guardar el modelo entrenado
with open('models/alzheimerModel.pkl', 'wb') as f:
    pickle.dump(bestModel, f)



