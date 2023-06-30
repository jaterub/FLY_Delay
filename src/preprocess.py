import pandas as pd
import numpy as np

# Leer datos en bruto
df = pd.read_csv('../data/raw/df_flight.csv')

# Eliminar columna 'Status'
df = df.drop('Status', axis=1)

# Crear columna 'Status' corregida
# Asignar valores de 0 a 2 según el valor de 'ARRIVAL_DELAY'
df.loc[df['ARRIVAL_DELAY'] <= 15, 'Status'] = 0  # leve
df.loc[(df['ARRIVAL_DELAY'] >= 15) & (
    df['ARRIVAL_DELAY'] < 60), 'Status'] = 1  # medio
df.loc[df['ARRIVAL_DELAY'] >= 60, 'Status'] = 2  # elevado

# Crear columna 'Status_label' mapeando valores de 'Status' a etiquetas
df['Status_label'] = df['Status'].map({0: 'leve', 1: 'medio', 2: 'elevado'})

# Crear columna 'DELAYED' con valores binarios según si hubo retraso o no
df['DELAYED'] = np.where(df['ARRIVAL_DELAY'] > 0, 1, 0)

# Eliminar columnas con más del 25% de valores faltantes
missing_columns = df.loc[:, df.isna().mean() >= 0.25].columns
df = df.drop(missing_columns, axis=1)

# Eliminar columnas innecesarias
df = df.drop(['YEAR', 'MONTH', 'FLIGHT_NUMBER',
             'TAIL_NUMBER', 'Status_label'], axis=1)

# Rellenar valores faltantes con la media de la columna correspondiente
remaining_na_columns = df.loc[:, df.isna().sum() > 0].columns
for column in remaining_na_columns:
    df[column] = df[column].fillna(df[column].mean())

# Función para codificar variables categóricas con one-hot encoding


def onehot_encode(df, column_dict):
    df = df.copy()
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


# Codificar variables categóricas con one-hot encoding
df = onehot_encode(df, column_dict={
                   'AIRLINE': 'AL', 'ORIGIN_AIRPORT': 'OA', 'DESTINATION_AIRPORT': 'DA'})

# Función para guardar dataframe en archivo csv


def guardar_en_csv(df, ruta_destino):
    df.to_csv(ruta_destino, index=False)
    print(f"DataFrame guardado en {ruta_destino}")


reduced_df = df.sample(frac=0.25)

# Guardar dataframe procesado en archivo csv
guardar_en_csv(df, '../data/processed/processed.csv')
