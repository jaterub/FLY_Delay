import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import yaml

# Read in processed data and split into train and test sets
df_processed = pd.read_csv('../data/processed/processed.csv')
train_df, test_df = train_test_split(
    df_processed, test_size=0.2, random_state=42)

# Function to save dataframes to csv files


def guardar_en_csv(df, ruta_destino):
    df.to_csv(ruta_destino, index=False)
    print(f"DataFrame guardado en {ruta_destino}")


# Save train and test dataframes to csv files
guardar_en_csv(train_df, '../data/train/train_df.csv')
guardar_en_csv(test_df, '../data/test/test_df.csv')

# Split data into features and target variable
X = df_processed.drop('DELAYED', axis=1).copy()
y = df_processed['DELAYED'].copy()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Definir el pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Escalador
    ('rf', RandomForestClassifier())  # Clasificador Random Forest
])

# Definir los hiperparámetros a ajustar
parameters = {
    'rf__n_estimators': [60, 70, 90],
    'rf__max_depth': [3],
    'rf__criterion': ['gini', 'log_loss']
}

# Realizar la búsqueda de hiperparámetros utilizando validación cruzada
grid_search = GridSearchCV(pipeline, parameters, cv=5,
                           scoring='accuracy', verbose=3)
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_model = grid_search.best_estimator_

print('Metricas Random Forest')

# Predecir con el mejor modelo en los datos de prueba
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

precision = precision_score(y_test, y_pred)
print("Precisión: {:.2f}%".format(precision * 100))

recall = recall_score(y_test, y_pred)
print("Recall: {:.2f}%".format(recall * 100))

f1 = f1_score(y_test, y_pred)
print("F1-score: {:.2f}%".format(f1 * 100))
confusion = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(confusion)

# Save trained model to pickle file
with open('../models/trained_model.pkl', 'wb') as archivo_salida:
    pickle.dump(best_model, archivo_salida)

# Save model configuration to YAML file
model_config = {
    'model_name': 'Logistic Regression',
    'best_params': best_model.get_params()
}
print("Modelo entrenado y guardado pkl")

with open('../models/model_config.yaml', 'w') as f:
    yaml.dump(model_config, f)
print("Modelo entrenado y guardado yaml")
