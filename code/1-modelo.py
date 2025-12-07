import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import shap

# ============================================================
# 0. CONFIGURACIÓN GENERAL
# ============================================================

SEED = 6967
np.random.seed(SEED)

print("\n=== INICIO PIPELINE F1 POSITION MODEL ===\n")

# ============================================================
# 1. CARGA DEL DATASET Y FILTRADO DE DNFs (CON MARGEN DE VUELTAS)
# ============================================================

datos = pd.read_csv("data/f1_pitstops_2018_2024.csv", sep=",")

print(">> Información inicial del dataset:")
datos.info()
print(f"Tamaño inicial del dataset: {datos.shape}\n")

# 1.1 Calcular vueltas máximas por (Season, Circuit)
vueltas_max = (
    datos
    .groupby(['Season', 'Circuit'])['Laps']
    .max()
    .rename('Max_Laps')
    .reset_index()
)

datos = datos.merge(vueltas_max, on=['Season', 'Circuit'], how='left')

# 1.2 Margen de vueltas permitido para pilotos doblados
MARGEN_VUELTAS = 1  # permitir estar a 1 vuelta del líder y seguir contando como finisher

mask_finished = datos['Laps'] >= (datos['Max_Laps'] - MARGEN_VUELTAS)

filas_antes = datos.shape[0]
datos = datos[mask_finished].copy()
filas_eliminadas = filas_antes - datos.shape[0]
print(f">> Filas eliminadas como DNFs reales (no terminan carrera): {filas_eliminadas}")
datos.drop(columns=['Max_Laps'], inplace=True)

print("\nDescripción estadística de las columnas numéricas (tras filtrar DNFs):")
print(datos.describe())
print("\nDescripción estadística de las columnas categóricas (tras filtrar DNFs):")
print(datos.describe(include='object'))

# ============================================================
# 2. ELIMINACIÓN DE COLUMNAS SOLO DE CONTEXTO
# ============================================================

print("\n=== 2) Eliminación de columnas de contexto ===")
datos.drop(
    columns=[
        "Location", "Country", "Date", "Round",
        "Race Name", "Time_of_race", "Total Pit Stops", "Fast Lap Attempts"
    ],
    inplace=True
)

print("Columnas restantes tras la eliminación:")
print(list(datos.columns))

# ============================================================
# 3. TRATAMIENTO ESPECÍFICO DE Pit_Lap Y CATEGÓRICAS
# ============================================================

print("\n=== 3) Tratamiento de Pit_Lap y variables categóricas ===")

# Aseguramos que los vacíos de Pit_Lap (cadenas vacías) se traten como NaN y pasen a numérico
datos['Pit_Lap'] = datos['Pit_Lap'].replace('', np.nan)
datos['Pit_Lap'] = pd.to_numeric(datos['Pit_Lap'], errors='coerce')

# Forzamos tipos categóricos y marcamos Unknown
datos['Tire Compound'] = datos['Tire Compound'].astype(str).replace('nan', 'Unknown')
datos['Pit_Time'] = datos['Pit_Time'].astype(str)

# Para el stint final (Pit_Time == "Final Stint"), los NaN de Pit_Lap pasan a -1
# Esto marca explícitamente “stint final sin vuelta de parada”
mask_final_stint = datos['Pit_Time'] == "Final Stint"
datos.loc[mask_final_stint, 'Pit_Lap'] = datos.loc[mask_final_stint, 'Pit_Lap'].fillna(-1)

print("\nCategorías en 'Tire Compound':")
print(datos['Tire Compound'].value_counts())
print("\nCategorías principales en 'Pit_Time':")
print(datos['Pit_Time'].value_counts().head(10))

# ============================================================
# 4. REVISIÓN DE TIPOS Y NULOS
# ============================================================

print("\n=== 4) Tipos de datos y nulos antes de imputar ===")
print("\nTipos de datos por columna:")
print(datos.dtypes)

num_cols = datos.select_dtypes(include=['int64', 'float64']).columns
cat_cols = datos.select_dtypes(include='object').columns

print("\nColumnas numéricas:", list(num_cols))
print("Columnas categóricas:", list(cat_cols))

print("\nNulos numéricos:")
print(datos[num_cols].isnull().sum()[lambda s: s > 0])
print("\nNulos categóricos:")
print(datos[cat_cols].isnull().sum()[lambda s: s > 0])

# (opcional) distribución de tipos reales por columna
for c in datos.columns:
    print(f"\nDistribución de tipos reales en la columna '{c}':")
    print(datos[c].apply(lambda x: type(x).__name__).value_counts())

# ============================================================
# 5. IMPUTACIÓN DE VALORES NUMÉRICOS
# ============================================================

print("\n=== 5) Imputación numérica (medianas) y Pit_Lap ===")

from sklearn.impute import SimpleImputer

# Columnas numéricas donde imputaremos por la mediana
mediana_cols = [
    'AvgPitStopTime', 'Air_Temp_C', 'Track_Temp_C', 'Humidity_%',
    'Wind_Speed_KMH', 'Lap Time Variation', 'Tire Usage Aggression',
    'Driver Aggression Score', 'Stint', 'Stint Length'
]

imp_mediana = SimpleImputer(strategy='median')
datos[mediana_cols] = imp_mediana.fit_transform(datos[mediana_cols])

# Imputación específica para Pit_Lap:
#  - mantenemos los -1 (stint final sin parada)
#  - para los NaN restantes, imputamos con la mediana de Pit_Lap en vueltas con parada
mask_nan_pit = datos['Pit_Lap'].isna()
mediana_pit = datos.loc[(~mask_nan_pit) & (datos['Pit_Lap'] != -1), 'Pit_Lap'].median()
datos.loc[mask_nan_pit, 'Pit_Lap'] = mediana_pit

print("\nNulos restantes tras la imputación numérica:")
print(datos.isnull().sum()[lambda s: s > 0])

# ============================================================
# 6. SPLIT TEMPORAL 2018–2023 (TRAIN) / 2024 (TEST)
# ============================================================

print("\n=== 6) Split temporal: Train 2018–2023, Test 2024 ===")

train_df = datos[datos['Season'].between(2018, 2023)].copy()
test_df  = datos[datos['Season'] == 2024].copy()

print(f"Filas para entrenar (2018-2023): {train_df.shape[0]}")
print(f"Filas para validar (2024): {test_df.shape[0]}")

# ============================================================
# 7. PREPARACIÓN DE FEATURES (NUMÉRICOS + CATEGÓRICOS)
# ============================================================

print("\n=== 7) Preparación de variables numéricas y categóricas ===")

# Variables numéricas que queremos usar
columnas_num = [
    'TotalPitStops', 'AvgPitStopTime', 'Air_Temp_C',
    'Track_Temp_C', 'Humidity_%', 'Wind_Speed_KMH',
    'Pit_Lap', 'Driver Aggression Score', 'Tire Usage Aggression'
]

# Variables categóricas
columnas_cat = ['Driver', 'Constructor', 'Tire Compound', 'Circuit']

# Target
y_train = train_df['Position']
y_test  = test_df['Position']

# 7.1 Numéricas (relleno de posibles NaN con medias de train)
X_train_num = train_df[columnas_num].copy()
X_test_num  = test_df[columnas_num].copy()
medias = X_train_num.mean()
X_train_num.fillna(medias, inplace=True)
X_test_num.fillna(medias, inplace=True)

# 7.2 Categóricas (one-hot encoding)
X_train_cat = pd.get_dummies(train_df[columnas_cat], columns=columnas_cat)
X_test_cat  = pd.get_dummies(test_df[columnas_cat], columns=columnas_cat)
X_test_cat  = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

# 7.3 Unión final
X_train_final = pd.concat([X_train_cat, X_train_num], axis=1)
X_test_final  = pd.concat([X_test_cat, X_test_num], axis=1)

print(f"Dimensiones finales TRAIN (filas, columnas): {X_train_final.shape}")
print(f"Dimensiones finales TEST  (filas, columnas): {X_test_final.shape}")

# ============================================================
# 8. MODELO 1 – RANDOM FOREST REGRESSOR
# ============================================================

print("\n=== 8) Modelo 1: RandomForestRegressor ===")

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=SEED,
    n_jobs=-1
)

rf.fit(X_train_final, y_train)
pred_rf = rf.predict(X_test_final)
pred_rf_round = np.round(pred_rf)
err_rf = np.abs(y_test - pred_rf_round)

mae_rf = mean_absolute_error(y_test, pred_rf)
acc_rf_exact = np.mean(err_rf == 0)
acc_rf_tol1 = np.mean(err_rf <= 1)

print("\n>> Resultados RandomForestRegressor (test 2024):")
print(f"MAE: {mae_rf:.2f}")
print(f"Acierto exacto: {acc_rf_exact:.2%}")
print(f"Acierto con margen ±1 puesto: {acc_rf_tol1:.2%}")

importancias_rf = pd.Series(rf.feature_importances_, index=X_train_final.columns)
print("\nTop 10 variables importantes (RandomForestRegressor):")
print(importancias_rf.sort_values(ascending=False).head(10))

# ============================================================
# 9. MODELO 2 – XGBOOST REGRESSOR
# ============================================================

print("\n=== 9) Modelo 2: XGBRegressor ===")

xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=SEED,
    n_jobs=-1
)

xgb.fit(X_train_final, y_train)
pred_xgb = xgb.predict(X_test_final)
pred_xgb_round = np.round(pred_xgb)
err_xgb = np.abs(y_test - pred_xgb_round)

mae_xgb = mean_absolute_error(y_test, pred_xgb)
acc_xgb_exact = np.mean(err_xgb == 0)
acc_xgb_tol1 = np.mean(err_xgb <= 1)  # margen ±1

print("\n>> Resultados XGBRegressor (test 2024):")
print(f"MAE: {mae_xgb:.2f}")
print(f"Acierto exacto: {acc_xgb_exact:.2%}")
print(f"Acierto con margen ±1 puesto: {acc_xgb_tol1:.2%}")

importancias_xgb = pd.Series(xgb.feature_importances_, index=X_train_final.columns)
print("\nTop 10 variables importantes (XGBRegressor):")
print(importancias_xgb.sort_values(ascending=False).head(10))

# ============================================================
# 10. MODELO 3 – HIST GRADIENT BOOSTING REGRESSOR
# ============================================================

print("\n=== 10) Modelo 3: HistGradientBoostingRegressor ===")

hgb = HistGradientBoostingRegressor(
    learning_rate=0.1,
    max_iter=100,
    random_state=SEED
)

hgb.fit(X_train_final, y_train)
pred_hgb = hgb.predict(X_test_final)
pred_hgb_round = np.round(pred_hgb)
err_hgb = np.abs(y_test - pred_hgb_round)

mae_hgb = mean_absolute_error(y_test, pred_hgb)
acc_hgb_exact = np.mean(err_hgb == 0)
acc_hgb_tol1 = np.mean(err_hgb <= 1)

print("\n>> Resultados HistGradientBoostingRegressor (test 2024):")
print(f"MAE: {mae_hgb:.2f}")
print(f"Acierto exacto: {acc_hgb_exact:.2%}")
print(f"Acierto con margen ±1 puesto: {acc_hgb_tol1:.2%}")

# ============================================================
# 11. ANÁLISIS DETALLADO DEL MEJOR MODELO (EJ: HGB)
# ============================================================

print("\n=== 11) Análisis avanzado del mejor modelo (ej. HistGB) ===")

resultados = pd.DataFrame({
    'Real': y_test,
    'Predicho': pred_hgb_round
})

errores_abs = np.abs(resultados['Real'] - resultados['Predicho'])
print("\nResumen de errores (HistGB):")
print(errores_abs.describe())

# Gráfico Predicción vs Realidad
plt.figure(figsize=(10, 6))
plt.scatter(resultados['Real'], resultados['Predicho'],
            alpha=0.6, c='blue', edgecolors='k', label='Predicciones')
plt.plot([1, 20], [1, 20], 'r--', linewidth=2, label='Predicción perfecta')
plt.fill_between([1, 20], [0, 19], [2, 21],
                 color='red', alpha=0.1, label='Margen ±1 puesto')

plt.title('Predicción vs Realidad (2024) - HistGradientBoosting', fontsize=14)
plt.xlabel('Posición REAL', fontsize=12)
plt.ylabel('Posición PREDICHA', fontsize=12)
plt.xticks(range(1, 21))
plt.yticks(range(1, 21))
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# ============================================================
# 12. IMPORTANCIA POR PERMUTACIÓN (sobre HGB)
# ============================================================

print("\n=== 12) Permutation Importance (HistGB) ===")
r = permutation_importance(
    hgb, X_train_final, y_train,
    n_repeats=10, random_state=SEED, n_jobs=-1
)

sorted_idx = r.importances_mean.argsort()[::-1][:10]

plt.figure(figsize=(10, 6))
plt.barh(X_train_final.columns[sorted_idx][::-1],
         r.importances_mean[sorted_idx][::-1],
         color='#2ca02c')
plt.title('Top 10 variables importantes (Permutation Importance, HistGB)', fontsize=14)
plt.xlabel('Impacto medio en el error', fontsize=12)
plt.tight_layout()
plt.show()

print("\nTop 10 factores (Permutation Importance):")
for i in sorted_idx:
    print(f"{X_train_final.columns[i]}: {r.importances_mean[i]:.4f}")

# ============================================================
# 13. ANÁLISIS SHAP (HistGB)
# ============================================================

print("\n=== 13) Análisis SHAP (HistGradientBoosting) ===")
explainer = shap.Explainer(hgb, X_train_final)
shap_values = explainer(X_test_final)

# Resumen global
plt.figure(figsize=(12, 8))
plt.title("SHAP summary plot - impacto de variables en Position")
shap.summary_plot(shap_values, X_test_final, show=False)
plt.show()

# Dependencia de AvgPitStopTime
shap.dependence_plot("AvgPitStopTime", shap_values.values, X_test_final,
                     interaction_index=None)
