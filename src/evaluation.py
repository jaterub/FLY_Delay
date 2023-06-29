import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el modelo entrenado desde el archivo pickle
with open('../models/trained_model.pkl', 'rb') as archivo_entrada:
    model = pickle.load(archivo_entrada)

# Cargar los datos de prueba
test_df = pd.read_csv('../data/test/test_df.csv')
X_test = test_df.drop('DELAYED', axis=1)
y_test = test_df['DELAYED']

# Realizar las predicciones utilizando el modelo
predictions = model.predict(X_test)

# Calcular la matriz de confusión
confusion_matrix = confusion_matrix(y_test, predictions)
print("Matriz de Confusión:")
print(confusion_matrix)

# Calcular el informe de clasificación
classification_report = classification_report(y_test, predictions)
print("Informe de Clasificación:")
print(classification_report)
