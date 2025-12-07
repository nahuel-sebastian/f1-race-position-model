import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor # Usamos Regresión para calcular distancias
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

import shap
from sklearn.inspection import permutation_importance
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


# ==========================================
# 5. NUEVO MODELO: HIST GRADIENT BOOSTING 🚀
# ==========================================
print("\n--- ENTRENANDO HIST GRADIENT BOOSTING ---")
# Este modelo es más potente que el Random Forest para regresión
modelo = HistGradientBoostingRegressor(
    learning_rate=0.1,
    max_iter=100,
    random_state=6967
)
modelo.fit(X_train_final, y_train)

# ==========================================
# 6. EVALUACIÓN AVANZADA (SIN ACCIDENTES)
# ==========================================
predicciones = modelo.predict(X_test_final)
predicciones_redondeadas = np.round(predicciones)

# Creamos un DataFrame para analizar los errores
resultados = pd.DataFrame({
    'Real': y_test,
    'Predicho': predicciones_redondeadas,
    'Paradas': test_df['TotalPitStops'], # Para detectar abandonos
    'AvgPitTime': test_df['AvgPitStopTime']
})

# FILTRO DE RENDIMIENTO PURO:
# Identificamos a los que terminaron la carrera (tienen tiempo de parada y paradas > 0 o lógica similar)
# Si AvgPitTime es NaN en el test, es que se retiraron pronto.
mask_finished = resultados['AvgPitTime'].notnull()

# Calculamos errores solo de los que terminaron
y_test_clean = results_clean = resultados[mask_finished]['Real']
pred_clean = resultados[mask_finished]['Predicho']
errores_clean = np.abs(y_test_clean - pred_clean)

mae_clean = mean_absolute_error(y_test_clean, pred_clean)
acc_margen = np.mean(errores_clean <= 1)

print(f"\nResultados en 2024 (Excluyendo Accidentes/Abandonos):")
print(f"1. Acierto con MARGEN (+/- 1 puesto): {acc_margen:.2%} 🎯")
print(f"2. Error Medio (MAE): {mae_clean:.2f} puestos")

# ==========================================
# 7. VISUALIZACIÓN DE PREDICCIÓN VS REALIDAD 📊
# ==========================================
plt.figure(figsize=(10, 6))

# Gráfico de dispersión
# Eje X: Posición Real, Eje Y: Posición Predicha
plt.scatter(resultados[mask_finished]['Real'], 
            resultados[mask_finished]['Predicho'], 
            alpha=0.6, c='blue', edgecolors='k', label='Predicciones')

# Línea diagonal perfecta (Si el punto cae aquí, es un acierto exacto)
plt.plot([1, 20], [1, 20], 'r--', linewidth=2, label='Predicción Perfecta')

# Bandas de margen de error (+/- 1 puesto)
plt.fill_between([1, 20], [0, 19], [2, 21], color='red', alpha=0.1, label='Margen Aceptable (+/- 1)')

plt.title(f'Predicción vs Realidad (2024) - Modelo Gradient Boosting', fontsize=14)
plt.xlabel('Posición REAL Final', fontsize=12)
plt.ylabel('Posición PREDICHA por el Modelo', fontsize=12)
plt.xticks(range(1, 21))
plt.yticks(range(1, 21))
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

# Importamos los 4 candidatos
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CARGA Y LIMPIEZA (La versión definitiva)
# ==========================================
print("--- PREPARANDO LA CARRERA DE MODELOS ---")
datos = pd.read_csv("data/f1_pitstops_2018_2024.csv", sep=",")

# Limpieza básica
datos.drop(columns=["Location", "Country", "Date", "Round", "Race Name", "Time_of_race", "Total Pit Stops", "Pit_Time", "Fast Lap Attempts"], inplace=True)
# Quitamos accidentes del histórico
datos.dropna(subset=['AvgPitStopTime'], inplace=True)

# División Temporal
train_df = datos[datos['Season'].between(2018, 2023)].copy()
test_df  = datos[datos['Season'] == 2024].copy()

# Variables
y_train = train_df['Position']
y_test  = test_df['Position']
columnas_num = ['TotalPitStops', 'AvgPitStopTime', 'Air_Temp_C', 'Track_Temp_C', 'Humidity_%', 'Wind_Speed_KMH', 'Driver Aggression Score', 'Tire Usage Aggression']
columnas_cat = ['Driver', 'Constructor', 'Circuit', 'Tire Compound']

# Preprocesado (Imputación + Encoding)
X_train_num = train_df[columnas_num].fillna(train_df[columnas_num].mean())
X_test_num  = test_df[columnas_num].fillna(train_df[columnas_num].mean())

X_train_cat = pd.get_dummies(train_df[columnas_cat], columns=columnas_cat)
X_test_cat  = pd.get_dummies(test_df[columnas_cat], columns=columnas_cat)
X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

X_train = pd.concat([X_train_cat, X_train_num], axis=1)
X_test  = pd.concat([X_test_cat, X_test_num], axis=1)

# Escalado (Necesario para SVR y Ridge)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[columnas_num] = scaler.fit_transform(X_train[columnas_num])
X_test_scaled[columnas_num] = scaler.transform(X_test[columnas_num])

# ==========================================
# 2. COMPETICIÓN DE MODELOS 🏁
# ==========================================
modelos = {
    "1. Regresión Lineal (Ridge)": Ridge(alpha=1.0),
    "2. Random Forest": RandomForestRegressor(n_estimators=100, random_state=6967, n_jobs=-1),
    "3. HistGradientBoosting": HistGradientBoostingRegressor(random_state=6967),
    "4. SVR (Support Vector)": SVR(C=1.0, epsilon=0.2),
    "5. XGBoost (Original)": XGBRegressor(  # <--- CONFIGURACIÓN XGBOOST
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=6967, 
        n_jobs=-1
    )
}

resultados = []

print(f"\n{'MODELO':<30} | {'MAE (Error Medio)':<15} | {'MARGEN +/- 1':<15}")
print("-" * 65)

for nombre, modelo in modelos.items():
    # Usamos datos escalados para Ridge y SVR, normales para árboles
    if "Ridge" in nombre or "SVR" in nombre:
        X_t, X_v = X_train_scaled, X_test_scaled
    else:
        X_t, X_v = X_train, X_test
    
    # Entrenar
    modelo.fit(X_t, y_train)
    
    # Predecir
    pred = modelo.predict(X_v)
    
    # Evaluar (Filtro anti-accidentes en Test)
    mask_clean = test_df['AvgPitStopTime'].notnull() # Solo los que acabaron
    y_real_clean = y_test[mask_clean]
    pred_clean = pred[mask_clean]
    
    mae = mean_absolute_error(y_real_clean, pred_clean)
    margen = np.mean(np.abs(y_real_clean - np.round(pred_clean)) <= 1)
    
    print(f"{nombre:<30} | {mae:.4f}          | {margen:.2%}")
    resultados.append({'Modelo': nombre, 'MAE': mae, 'Margen': margen})

# ==========================================
# 3. EL GANADOR 🏆
# ==========================================
mejor_modelo = min(resultados, key=lambda x: x['MAE'])
print(f"\n🏆 GANADOR DE LA SESIÓN: {mejor_modelo['Modelo']}")
print(f"   -> Se equivoca solo por {mejor_modelo['MAE']:.2f} posiciones de media.")
print(f"   -> Acierta el puesto exacto (o casi) el {mejor_modelo['Margen']:.1%} de las veces.")

# ==========================================
# 3. GRÁFICO 1: PREDICCIÓN VS REALIDAD
# ==========================================
predicciones = modelo.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicciones, alpha=0.5, color='#1f77b4', label='Predicciones')
plt.plot([1, 20], [1, 20], 'r--', lw=2, label='Perfecto') # Línea diagonal
plt.fill_between([1, 20], [0, 19], [2, 21], color='red', alpha=0.1, label='Margen +/- 1')

plt.title('Rendimiento del Modelo: Predicción vs Realidad (2024)', fontsize=14)
plt.xlabel('Posición REAL', fontsize=12)
plt.ylabel('Posición PREDICHA', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 4. GRÁFICO 2: IMPORTANCIA DE VARIABLES (PERMUTACIÓN)
# ==========================================
print("Calculando importancia de variables (esto tarda unos segundos)...")
# Usamos Permutation Importance porque HistGradientBoosting no tiene feature_importances_ nativo
r = permutation_importance(modelo, X_train, y_train, n_repeats=10, random_state=6967, n_jobs=-1)

# Ordenamos
sorted_idx = r.importances_mean.argsort()[::-1][:10] # Top 10

plt.figure(figsize=(10, 6))
plt.barh(X_train.columns[sorted_idx][::-1], r.importances_mean[sorted_idx][::-1], color='#2ca02c')
plt.title('Top 10 Variables más Importantes (Permutation Importance)', fontsize=14)
plt.xlabel('Impacto en la reducción del error', fontsize=12)
plt.tight_layout()
plt.show()

# Imprimir valores numéricos
print("\n--- TOP FACTORES ---")
for i in sorted_idx:
    print(f"{X_train.columns[i]}: {r.importances_mean[i]:.4f}")


print("\nCalculando impacto de estrategias (SHAP)...")

explainer = shap.Explainer(modelo)
shap_values = explainer(X_test)

# GRÁFICO 1: Resumen General (Beeswarm)
plt.figure(figsize=(12, 8))
plt.title("¿Qué variables bajan (mejoran) la posición final?")
shap.summary_plot(shap_values, X_test, show=False)
plt.show()

# GRÁFICO 2: Dependencia (Ejemplo: Tiempo de Parada vs Posición)
# Esto te dice: "¿A partir de cuántos segundos pierdo una posición?"
shap.dependence_plot("AvgPitStopTime", shap_values.values, X_test, interaction_index=None)