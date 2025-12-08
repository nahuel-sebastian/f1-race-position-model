import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import shap

# ============================================================
# 0. CONFIGURACIÓN GENERAL
# ============================================================

# Semilla comun para distribucion de los datos
SEED = 6967
np.random.seed(SEED)

print("\n=== 1. CARGA DEL DATASET INICIAL ===\n")

# ============================================================
# 1. CARGA DEL DATASET Y DESCRIPCION INICIAL DE LOS DATOS
# ============================================================

# Extraemos los datos del CSV
datos = pd.read_csv("data/f1_pitstops_2018_2024.csv", sep=",")

# Informacion inicial del dataset
print(">> Información inicial del dataset:")
datos.info()
print(f"Tamaño inicial del dataset: {datos.shape}\n")

# Variables numericas y categoricas
print("\nDescripción estadística de las columnas numéricas:")
print(datos.describe())
print("\nDescripción estadística de las columnas categóricas:")
print(datos.describe(include='object'))

# ============================================================
# 2. PREPROCESADO DE DATOS SEGUN CONTEXTO
# ============================================================

print("\n=== 2. ELIMINACION DE COLUMNAS SEGUN CONTEXTO ===")

# Vueltas máximas por Temporada y Circuito
vueltas_max = (
    datos
    .groupby(['Season', 'Circuit'])['Laps']
    .max()
    .rename('Max_Laps')
    .reset_index()
)
datos = datos.merge(vueltas_max, on=['Season', 'Circuit'], how='left')

# Margen de vueltas permitido para pilotos doblados
MARGEN_VUELTAS = 1
mask_finished = datos['Laps'] >= (datos['Max_Laps'] - MARGEN_VUELTAS)

# Eliminacion de filas que representan ruido
filas_antes = datos.shape[0]
datos = datos[mask_finished].copy()
filas_eliminadas = filas_antes - datos.shape[0]
print(f">> Filas eliminadas como DNFs reales (no terminan carrera): {filas_eliminadas}")
datos.drop(columns=['Max_Laps'], inplace=True)

# Seleccion de atributos mas relevantes para la obtencion del resultado del modelo
datos.drop(
    columns=[
        "Location", "Country", "Date", "Round",
        "Race Name", "Time_of_race", "Total Pit Stops", "Fast Lap Attempts"
    ],
    inplace=True
)
print("Columnas restantes tras la eliminación:")
print(list(datos.columns))

# Variables numericas y categoricas tras la eliminacion de filas Y columnas
print("\nDescripción estadística de las columnas numéricas (tras filtrar DNFs):")
print(datos.describe())
print("\nDescripción estadística de las columnas categóricas (tras filtrar DNFs):")
print(datos.describe(include='object'))

# ============================================================
# 3. TRATAMIENTO DE TIPADO DE DATOS
# ============================================================

print("\n=== 3. NORMALIZACION DE TIPOS DE DATOS ===")

# Comprobación detallada del tipo real de los valores en cada columna
for c in datos.columns:
    print(f"\nDistribución de tipos reales en la columna '{c}':")
    print(datos[c].apply(lambda x: type(x).__name__).value_counts())

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
# 4. IDENTIFICACIÓN DE VARIABLES Y NULOS
# ============================================================

num_cols_global = datos.select_dtypes(include=['int64', 'float64']).columns
cat_cols_global = datos.select_dtypes(include='object').columns
print("\nColumnas numéricas (global):", list(num_cols_global))
print("Columnas categóricas (global):", list(cat_cols_global))

print("\nNulos numéricos:")
print(datos[num_cols_global].isnull().sum()[lambda s: s > 0])
print("\nNulos categóricos:")
print(datos[cat_cols_global].isnull().sum()[lambda s: s > 0])

# ============================================================
# 5. SPLIT TEMPORAL 2018–2023 (TRAIN) / 2024 (TEST)
# ============================================================

print("\n=== 6. SPLIT TEMPORAL: TRAIN 2018–2023, TEST 2024 ===")

train_df = datos[datos['Season'].between(2018, 2023)].copy()
test_df  = datos[datos['Season'] == 2024].copy()

print(f"Filas para entrenar (2018-2023): {train_df.shape[0]}")
print(f"Filas para validar (2024): {test_df.shape[0]}")

# ============================================================
# 6. IMPUTACION Y ESCALADO DE LA PARTICION DE ENTRENAMIENTO
# ============================================================

print("\n=== 6. IMPUTACION Y ESCALADO DE LA PARTICION DE ENTRENAMIENTO ===")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

# Features del modelo
num_cols = [
    'TotalPitStops', 'AvgPitStopTime', 'Air_Temp_C',
    'Track_Temp_C', 'Humidity_%', 'Wind_Speed_KMH',
    'Pit_Lap', 'Driver Aggression Score', 'Tire Usage Aggression'
]
cat_cols = ['Driver', 'Constructor', 'Tire Compound', 'Circuit']

# X e y SOLO con datos de entrenamiento (2018–2023)
X_train = train_df[num_cols + cat_cols].copy()
y_train = train_df['Position']

# Listas de escaladores e imputadores numéricos a probar
scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]
imputers = [SimpleImputer(strategy="mean"), SimpleImputer(strategy="median")]
mejor_scaler = None
mejor_imputer = None
mejor_score = np.inf

# Transformer para columnas categóricas
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

for scaler in scalers:
    for imputer in imputers:
        # Pipeline para columnas numéricas
        numerical_transformer = Pipeline(steps=[
            ("imputer", imputer),
            ("scaler", scaler)
        ])

        # Preprocesador combinado
        preprocessor = ColumnTransformer(transformers=[
            ("num", numerical_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ])

        # Pipeline completo con KNN REGRESOR
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", KNeighborsRegressor(n_neighbors=5))
        ])

        # 3-fold CV, métrica MAE (negativa)
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )
        mae_media = -scores.mean()

        print(f"Scaler: {scaler}, Imputer: {imputer}, MAE medio: {mae_media:.4f}")

        if mae_media < mejor_score:
            mejor_score = mae_media
            mejor_scaler = scaler
            mejor_imputer = imputer

print(f"\nMejor Scaler: {mejor_scaler}, Mejor Imputer: {mejor_imputer}, Mejor MAE: {mejor_score:.4f}")

# ============================================================
# 7. PREPROCESADOR DEFINITIVO (PARA TODOS LOS MODELOS)
# ============================================================

print("\n=== 7. PREPROCESADOR DEFINITIVO ===")

# 1) Codificador de variables categóricas
categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 2) Mejor escalador e imputador obtenidos con KNN
numerical_transformer = Pipeline(steps=[
    ("imputer", mejor_imputer),
    ("scaler",  mejor_scaler)
])

# 3) Preprocesador combinado
preprocesador = ColumnTransformer(transformers=[
    ("num", numerical_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

print("Preprocesador definido con:")
print(f"Imputador numérico: {mejor_imputer}")
print(f"Escalador numérico: {mejor_scaler}")
print(f"Columnas numéricas: {num_cols}")
print(f"Columnas categóricas: {cat_cols}")

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
plt.savefig("plots/hgb_pred_vs_real_2024.png")
plt.close()

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
plt.savefig("plots/hgb_permutation_importance_top10.png")
plt.close()

print("\nTop 10 factores (Permutation Importance):")
for i in sorted_idx:
    print(f"{X_train_final.columns[i]}: {r.importances_mean[i]:.4f}")

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# ============================================================
# 13. AJUSTE DE HIPERPARÁMETROS (RF, XGB, HGB)
# ============================================================

print("\n=== 13) Búsqueda de hiperparámetros (RandomizedSearchCV, CV=3, métrica MAE) ===")

# ------------ 13.1 Random Forest ------------
print("\n>>> 13.1 Ajustando RandomForestRegressor...")

param_rf = {
    "n_estimators": randint(200, 800),
    "max_depth": randint(3, 20),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", 0.5, 0.7]
}

rf_base = RandomForestRegressor(random_state=SEED, n_jobs=-1)

rf_search = RandomizedSearchCV(
    rf_base,
    param_distributions=param_rf,
    n_iter=30,                      # ajusta según tiempo disponible
    scoring="neg_mean_absolute_error",
    cv=3,
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train_final, y_train)
print("\nMejores hiperparámetros RF:", rf_search.best_params_)
rf_tuned = rf_search.best_estimator_

# Re‑evaluación en 2024
pred_rf_tuned = rf_tuned.predict(X_test_final)
pred_rf_tuned_round = np.round(pred_rf_tuned)
err_rf_tuned = np.abs(y_test - pred_rf_tuned_round)

mae_rf_tuned = mean_absolute_error(y_test, pred_rf_tuned)
acc_rf_tuned_exact = np.mean(err_rf_tuned == 0)
acc_rf_tuned_tol1 = np.mean(err_rf_tuned <= 1)

print("\n>> RF AJUSTADO (test 2024):")
print(f"MAE: {mae_rf_tuned:.2f}")
print(f"Acierto exacto: {acc_rf_tuned_exact:.2%}")
print(f"Acierto con margen ±1 puesto: {acc_rf_tuned_tol1:.2%}")

# ------------ 13.2 XGBoost ------------
print("\n>>> 13.2 Ajustando XGBRegressor...")

param_xgb = {
    "n_estimators": randint(300, 900),
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.19),   # 0.01–0.20
    "subsample": uniform(0.6, 0.4),         # 0.6–1.0
    "colsample_bytree": uniform(0.6, 0.4),  # 0.6–1.0
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0.0, 0.5),
    "reg_lambda": uniform(0.0, 2.0)
}

xgb_base = XGBRegressor(
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist"
)

xgb_search = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_xgb,
    n_iter=40,
    scoring="neg_mean_absolute_error",
    cv=3,
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train_final, y_train)
print("\nMejores hiperparámetros XGB:", xgb_search.best_params_)
xgb_tuned = xgb_search.best_estimator_

pred_xgb_tuned = xgb_tuned.predict(X_test_final)
pred_xgb_tuned_round = np.round(pred_xgb_tuned)
err_xgb_tuned = np.abs(y_test - pred_xgb_tuned_round)

mae_xgb_tuned = mean_absolute_error(y_test, pred_xgb_tuned)
acc_xgb_tuned_exact = np.mean(err_xgb_tuned == 0)
acc_xgb_tuned_tol1 = np.mean(err_xgb_tuned <= 1)

print("\n>> XGB AJUSTADO (test 2024):")
print(f"MAE: {mae_xgb_tuned:.2f}")
print(f"Acierto exacto: {acc_xgb_tuned_exact:.2%}")
print(f"Acierto con margen ±1 puesto: {acc_xgb_tuned_tol1:.2%}")

# ------------ 13.3 HistGradientBoosting ------------
print("\n>>> 13.3 Ajustando HistGradientBoostingRegressor...")

param_hgb = {
    "max_iter": randint(150, 700),
    "learning_rate": uniform(0.01, 0.19),
    "max_depth": randint(3, 15),
    "min_samples_leaf": randint(5, 50),
    "l2_regularization": uniform(0.0, 2.0),
    "max_leaf_nodes": randint(15, 63)
}

hgb_base = HistGradientBoostingRegressor(
    random_state=SEED,
    early_stopping=True
)

hgb_search = RandomizedSearchCV(
    hgb_base,
    param_distributions=param_hgb,
    n_iter=40,
    scoring="neg_mean_absolute_error",
    cv=3,
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)

hgb_search.fit(X_train_final, y_train)
print("\nMejores hiperparámetros HGB:", hgb_search.best_params_)
hgb_tuned = hgb_search.best_estimator_

pred_hgb_tuned = hgb_tuned.predict(X_test_final)
pred_hgb_tuned_round = np.round(pred_hgb_tuned)
err_hgb_tuned = np.abs(y_test - pred_hgb_tuned_round)

mae_hgb_tuned = mean_absolute_error(y_test, pred_hgb_tuned)
acc_hgb_tuned_exact = np.mean(err_hgb_tuned == 0)
acc_hgb_tuned_tol1 = np.mean(err_hgb_tuned <= 1)

print("\n>> HGB AJUSTADO (test 2024):")
print(f"MAE: {mae_hgb_tuned:.2f}")
print(f"Acierto exacto: {acc_hgb_tuned_exact:.2%}")
print(f"Acierto con margen ±1 puesto: {acc_hgb_tuned_tol1:.2%}")

# ============================================================
# 14. ANÁLISIS SHAP (XGBoost)
# ============================================================

print("\n=== 13) Análisis SHAP (XGBRegressor) ===")

# Nos aseguramos de que los datos sean float32 (sin dtype object)
X_train_shap = X_train_final.astype(np.float32)
X_test_shap  = X_test_final.astype(np.float32)

explainer = shap.TreeExplainer(xgb)
shap_values = explainer(X_test_shap)

# Resumen global
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_shap, show=False)
plt.title("SHAP summary plot - impacto de variables en Position (XGBoost)")
plt.tight_layout()
plt.savefig("plots/shap_summary_xgb.png")
plt.close()

# Dependencia de AvgPitStopTime
shap.dependence_plot("AvgPitStopTime", shap_values.values, X_test_shap,
                     interaction_index=None, show=False)
plt.tight_layout()
plt.savefig("plots/shap_dependence_AvgPitStopTime_xgb.png")
plt.close()
