from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Hacer predicciones en los datos de prueba con cada modelo
y_pred_svc1_model = svc1_model.predict(X_test)
y_pred_gb1_model = gb1_model.predict(X_test)
y_pred_kmeans4_model = kmeans4_model.predict(X_test)

# Calcular las métricas de rendimiento para cada modelo
# f1_svc1
accuracy_svc1 = accuracy_score(y_test, y_pred_svc1_model)
precision_svc1 = precision_score(y_test, y_pred_svc1_model, average='weighted')
recall_svc1 = recall_score(y_test, y_pred_svc1_model, average='weighted')
f1_svc1 = f1_score(y_test, y_pred_svc1_model, average='weighted')

# accuracy_gb1
accuracy_gb1 = accuracy_score(y_test, y_pred_gb1_model)
precision_gb1 = precision_score(y_test, y_pred_gb1_model, average='weighted')
recall_gb1 = recall_score(y_test, y_pred_gb1_model, average='weighted')
f1_gb1 = f1_score(y_test, y_pred_gb1_model, average='weighted')

# accuracy_kmeans4
accuracy_kmeans4 = accuracy_score(y_test, y_pred_kmeans4_model)
precision_kmeans4 = precision_score(
    y_test, y_pred_kmeans4_model, average='weighted')
recall_kmeans4 = recall_score(y_test, y_pred_kmeans4_model, average='weighted')
f1_kmeans4 = f1_score(y_test, y_pred_kmeans4_model, average='weighted')

# Imprimir las métricas de rendimiento para cada modelo
print('Métricas de rendimiento para svc1_model:')
print(f'Accuracy: {accuracy_svc1:.2f}')
print(f'Precision: {precision_svc1:.2f}')
print(f'Recall: {recall_svc1:.2f}')
print(f'F1-score: {f1_svc1:.2f}')

print('Métricas de rendimiento para gb1_model:')
print(f'Accuracy: {accuracy_gb1:.2f}')
print(f'Precision: {precision_gb1:.2f}')
print(f'Recall: {recall_gb1:.2f}')
print(f'F1-score: {f1_gb1:.2f}')

print('Métricas de rendimiento para kmeans4_model:')
print(f'Accuracy: {accuracy_kmeans4:.2f}')
print(f'Precision: {precision_kmeans4:.2f}')
print(f'Recall: {recall_kmeans4:.2f}')
print(f'F1-score: {f1_kmeans4:.2f}')
