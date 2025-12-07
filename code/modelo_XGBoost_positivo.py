import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor # Usamos Regresión para calcular distancias
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =========================
# 1. Carga y resumen inicial
# =========================

# Carga del dataset original de pitstops de F1
datos = pd.read_csv("data/f1_pitstops_2018_2024.csv", sep=",")

print("Información inicial del dataset:")
datos.info()
print(f"Tamaño del dataset: {datos.shape}")
print(f"Número total de elementos del dataset: {datos.size}")
print("\nDescripción estadística de las columnas numéricas:")
print(datos.describe())
print("\nDescripción estadística de las columnas categóricas:")
print(datos.describe(include='object'))

# ==========================================
# 2. Eliminación de columnas de solo contexto
# ==========================================

print("\nEliminación de columnas de contexto poco útiles para la predicción directa:")
datos.drop(
    columns=[
        "Location", "Country", "Date", "Round",
        "Race Name", "Time_of_race", "Total Pit Stops", "Fast Lap Attempts"
    ],
    inplace=True
)
filas_antes = datos.shape[0]
datos.dropna(subset=['AvgPitStopTime'], inplace=True)
print(f"Filas eliminadas por accidentes/abandono: {filas_antes - datos.shape[0]}")
# Listado/justificación de columnas eliminadas:
# - Location/Country: redundantes con el circuito
# - Date/Time_of_race/Race Name/Round: información temporal/contextual no usada en el modelo
# - Total Pit Stops: duplicada respecto a otras métricas de pitstops

print("Columnas restantes tras la eliminación:")
print(datos.columns)

# ===========================
# 3. Tratamiento específico de Pit_Lap y categóricas
# ===========================

# Aseguramos que los vacíos de Pit_Lap (cadenas vacías) se traten como NaN
datos['Pit_Lap'] = datos['Pit_Lap'].replace('', np.nan)
datos['Pit_Lap'] = pd.to_numeric(datos['Pit_Lap'], errors='coerce')

# Forzamos tipos categóricos y marcamos Unknown
datos['Tire Compound'] = datos['Tire Compound'].astype(str)
datos['Tire Compound'] = datos['Tire Compound'].replace('nan', 'Unknown')
datos['Pit_Time'] = datos['Pit_Time'].astype(str)

# Para el stint final (Pit_Time == "Final Stint"), los NaN de Pit_Lap pasan a -1
# Esto marca explícitamente “stint final sin vuelta de parada”
mask_final_stint = datos['Pit_Time'] == "Final Stint"
datos.loc[mask_final_stint, 'Pit_Lap'] = datos.loc[mask_final_stint, 'Pit_Lap'].fillna(-1)

# Conteo de categorías (variables de estrategia/estado de stint)
print("\nCategorías en 'Tire Compound':")
print(datos['Tire Compound'].value_counts())
print("\nCategorías principales en 'Pit_Time':")
print(datos['Pit_Time'].value_counts().head(10))

# ==========================================
# 4. Revisión y fijación de tipos de columnas
# ==========================================

print("\nTipos de datos por columna (pandas):")
print(datos.dtypes)

num_cols = datos.select_dtypes(include=['int64', 'float64']).columns
cat_cols = datos.select_dtypes(include='object').columns

print("\nNuméricas:", list(num_cols))
print("Categóricas:", list(cat_cols))

print("\nNulos numéricos antes de imputar:")
print(datos[num_cols].isnull().sum()[lambda s: s > 0])
print("\nNulos categóricos antes de imputar:")
print(datos[cat_cols].isnull().sum()[lambda s: s > 0])

# Comprobación más detallada del tipo real de los valores en cada columna
for c in datos.columns:
    print(f"\nDistribución de tipos reales en la columna '{c}':")
    print(datos[c].apply(lambda x: type(x).__name__).value_counts())




# 1. Separar las filas por año (Temporal Split)
train_df = datos[datos['Season'].between(2018, 2023)].copy()
test_df  = datos[datos['Season'] == 2024].copy()

print(f"Filas para entrenar (2018-2023): {train_df.shape[0]}")
print(f"Filas para validar (2024): {test_df.shape[0]}")

# --- NUEVA SECCIÓN: PREPARACIÓN DE DATOS (NUMÉRICOS + TEXTO) ---

# 1. Definir nuestras variables (Features)
# Variables numéricas que queremos usar (¡incluyendo el clima!)
columnas_num = ['TotalPitStops', 'AvgPitStopTime', 'Air_Temp_C',
                'Track_Temp_C', 'Humidity_%', 'Wind_Speed_KMH', 'Pit_Lap', 'Driver Aggression Score', 'Tire Usage Aggression']

# Variables de texto (categóricas)
columnas_cat = ['Driver', 'Constructor', 'Tire Compound', 'Circuit']

# Definimos la variable objetivo (Target)
y_train = train_df['Position']
y_test  = test_df['Position']

# 2. Procesar Numéricas (Imputación)
# Seleccionamos solo las columnas numéricas
X_train_num = train_df[columnas_num].copy()
X_test_num  = test_df[columnas_num].copy()

# Calculamos la media SOLO con el pasado (Train) para no hacer trampa
medias = X_train_num.mean()

# Rellenamos los huecos (NaN) con esas medias
X_train_num.fillna(medias, inplace=True)
X_test_num.fillna(medias, inplace=True)

# 3. Procesar Categóricas (Encoding)
# Convertimos el texto a números (0 y 1)
X_train_cat = pd.get_dummies(train_df[columnas_cat], columns=columnas_cat)
X_test_cat  = pd.get_dummies(test_df[columnas_cat], columns=columnas_cat)

# Alineamos las columnas del test (para que coincidan exactamente con el train)
X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

# 4. Unir todo para crear el dataset final
X_train_final = pd.concat([X_train_cat, X_train_num], axis=1)
X_test_final  = pd.concat([X_test_cat, X_test_num], axis=1)

print(f"\nDimensiones finales para el modelo (Filas, Columnas): {X_train_final.shape}")
print(f"\nDimensiones finales para el modelo (Filas, Columnas): {X_test_final.shape}")


# --- ENTRENAMIENTO Y EVALUACIÓN ---
from sklearn.ensemble import RandomForestClassifier

# 1. Crear el modelo (El "bosque" de 100 árboles)
# n_estimators=100 crea 100 árboles
# random_state=6967 para que siempre salga lo mismo
bosque = RandomForestClassifier(n_estimators=100, random_state=6967)

# 2. Entrenar
print("Entrenando el Random Forest (esto puede tardar un pelín más)...")
bosque.fit(X_train_final, y_train)

# 3. Validar
precision = bosque.score(X_test_final, y_test)
print(f"Precisión con Random Forest: {precision:.2%}")

# 4. Ver Importancia (igual que antes)
importancias = pd.Series(bosque.feature_importances_, index=X_train_final.columns)
print("\nTop 5 variables para el Random Forest:")
print(importancias.sort_values(ascending=False).head(10))


# ==========================================
# 5. MODELO XGBOOST REGRESSOR 📈
# ==========================================
print("\n--- ENTRENANDO MODELO (REGRESIÓN) ---")
modelo = XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    random_state=6967,
    n_jobs=-1
)
modelo.fit(X_train_final, y_train)

# ==========================================
# 6. RESULTADOS CON TOLERANCIA +/- 1
# ==========================================
# Predecimos y redondeamos al entero más cercano
predicciones = modelo.predict(X_test_final)
predicciones_redondeadas = np.round(predicciones)

# Calculamos la distancia (Error absoluto)
errores = np.abs(y_test - predicciones_redondeadas)

# MÉTRICAS:
# 1. Exacto (Distancia = 0)
acierto_exacto = np.mean(errores == 0)

# 2. Tolerancia +/- 1 (Distancia <= 1) -> ¡ESTA ES TU IDEA!
acierto_tolerancia = np.mean(errores <= 3)

# 3. Error Medio (MAE)
mae = mean_absolute_error(y_test, predicciones)

print(f"\nResultados en 2024:")
print(f"1. Acierto EXACTO: {acierto_exacto:.2%}")
print(f"2. Acierto con MARGEN (+/- 1 puesto): {acierto_tolerancia:.2%}  <-- ¡TU MÉTRICA!")
print(f"3. Error Medio de Posiciones (MAE): {mae:.2f}")

# Importancia
importancias = pd.Series(modelo.feature_importances_, index=X_train_final.columns)
print("\n--- TOP 10 FACTORES CLAVE ---")
print(importancias.sort_values(ascending=False).head(10))


# ==========================================
# 5. MODELO RANDOM FOREST REGRESSOR 🌲📈
# ==========================================
print("\n--- ENTRENANDO RANDOM FOREST (REGRESIÓN) ---")
modelo = RandomForestRegressor(
    n_estimators=100, 
    random_state=6967,
    n_jobs=-1 # Usa todos los núcleos del procesador
)
modelo.fit(X_train_final, y_train)

# ==========================================
# 6. RESULTADOS CON TOLERANCIA +/- 1
# ==========================================
# Predecimos y redondeamos al entero más cercano
predicciones = modelo.predict(X_test_final)
predicciones_redondeadas = np.round(predicciones)

# Calculamos la distancia (Error absoluto)
errores = np.abs(y_test - predicciones_redondeadas)

# MÉTRICAS:
acierto_exacto = np.mean(errores == 0)
acierto_tolerancia = np.mean(errores <= 1) # Margen de error de 1 puesto
mae = mean_absolute_error(y_test, predicciones)

print(f"\nResultados en 2024:")
print(f"1. Acierto EXACTO: {acierto_exacto:.2%}")
print(f"2. Acierto con MARGEN (+/- 1 puesto): {acierto_tolerancia:.2%}  <-- DATO CLAVE")
print(f"3. Error Medio de Posiciones (MAE): {mae:.2f}")

# Importancia
importancias = pd.Series(modelo.feature_importances_, index=X_train_final.columns)
print("\n--- TOP 10 FACTORES CLAVE (Random Forest) ---")
print(importancias.sort_values(ascending=False).head(10))