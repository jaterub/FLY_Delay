import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Create pipeline with scaler and logistic regression model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])

# Define hyperparameters to search over
parameters = {
    'lr__C': [0.1, 1, 0.01],
    'lr__penalty': ['l2']
}

# Perform grid search with cross-validation to find best hyperparameters
grid_search = GridSearchCV(pipeline, parameters, cv=5,
                           scoring='recall', verbose=2)
grid_search.fit(X_train, y_train)

# Get best model from grid search
best_model = grid_search.best_estimator_

print('Metricas Logistic Regression')

# Make predictions with best model on test data
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("accuracy: {:.2f}%".format(accuracy * 100))
print("precision: {:.2f}%".format(precision * 100))
print("recall: {:.2f}%".format(recall * 100))
print("F1 score: {:.2f}%".format(f1 * 100))

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
