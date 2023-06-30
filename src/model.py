import pickle

# Cargar el modelo entrenado desde un archivo pickle
with open('../models/gb1_model.pkl', 'rb') as archivo_entrada:
    gb1_model = pickle.load(archivo_entrada)
