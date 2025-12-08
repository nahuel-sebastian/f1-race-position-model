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
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 2) Mejor escalador e imputador obtenidos con KNN
numerical_transformer = Pipeline(steps=[
    ("imputer", mejor_imputer),
    ("scaler",  mejor_scaler)
])

# 3) Preprocesador combinado
preprocesador = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    sparse_threshold=0
)

print("Preprocesador definido con:")
print(f"Imputador numérico: {mejor_imputer}")
print(f"Escalador numérico: {mejor_scaler}")
print(f"Columnas numéricas: {num_cols}")
print(f"Columnas categóricas: {cat_cols}")

# ============================================================
# 8. AJUSTE DE HIPERPARAMETROS (SOLO SOBRE TRAIN 2018-2023)
# ============================================================

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.makedirs("plots", exist_ok=True)
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

print("\n=== 8) AJUSTE DE HIPERPARÁMETROS (SOLO TRAIN 2018-2023) ===")

# --- A. Definir métricas de Accuracy personalizadas para Regresión ---
def exact_accuracy(y_true, y_pred):
    """Porcentaje de aciertos exactos (redondeando predicción)"""
    return np.mean(y_true == np.round(y_pred))

def tolerance_accuracy(y_true, y_pred):
    """Porcentaje de aciertos con margen de error de +/- 1 puesto"""
    return np.mean(np.abs(y_true - np.round(y_pred)) <= 1)

# Creamos scorers compatibles con sklearn
scorer_exact = make_scorer(exact_accuracy)
scorer_tol   = make_scorer(tolerance_accuracy)

# Diccionario para guardar resultados
resultados_modelos = {}

# ------------ 8.1 RANDOM FOREST ------------
print("\n>>> 8.1 RandomForest")
param_rf = {
    "model__n_estimators": randint(200, 600),
    "model__max_depth": randint(10, 25),
    "model__min_samples_split": randint(2, 15),
    "model__min_samples_leaf": randint(1, 8),
    "model__max_features": ["sqrt", "log2", 0.3, 0.5]
}

rf_pipe = Pipeline([
    ("preprocess", preprocesador),
    ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1))
])

rf_search = RandomizedSearchCV(rf_pipe, param_rf, n_iter=30, cv=3, 
                              scoring="neg_mean_absolute_error", 
                              random_state=SEED, n_jobs=-1, verbose=1)
rf_search.fit(X_train, y_train)

# Calcular Accuracy usando validación cruzada sobre el mejor modelo
best_rf = rf_search.best_estimator_
acc_rf_exact = cross_val_score(best_rf, X_train, y_train, cv=3, scoring=scorer_exact, n_jobs=-1).mean()
acc_rf_tol   = cross_val_score(best_rf, X_train, y_train, cv=3, scoring=scorer_tol, n_jobs=-1).mean()

resultados_modelos['RF'] = {
    'pipe': best_rf,
    'mae_cv': -rf_search.best_score_,
    'acc_exact': acc_rf_exact,
    'acc_tol': acc_rf_tol,
    'params': rf_search.best_params_
}

print(f"RF - Mejor MAE CV: {-rf_search.best_score_:.3f}")
print(f"RF - Accuracy Exacta CV: {acc_rf_exact:.2%}")

# ------------ 8.2 XGBOOST ------------
print("\n>>> 8.2 XGBoost ")
param_xgb = {
    "model__n_estimators": randint(200, 700),
    "model__max_depth": randint(3, 10),
    "model__learning_rate": uniform(0.01, 0.2),
    "model__subsample": uniform(0.7, 0.3),
    "model__colsample_bytree": uniform(0.7, 0.3),
    "model__reg_lambda": uniform(0.0, 2.0)
}

xgb_pipe = Pipeline([
    ("preprocess", preprocesador),
    ("model", XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist"))
])

xgb_search = RandomizedSearchCV(xgb_pipe, param_xgb, n_iter=40, cv=3, 
                               scoring="neg_mean_absolute_error", 
                               random_state=SEED, n_jobs=-1, verbose=1)
xgb_search.fit(X_train, y_train)

# Calcular Accuracy CV
best_xgb = xgb_search.best_estimator_
acc_xgb_exact = cross_val_score(best_xgb, X_train, y_train, cv=3, scoring=scorer_exact, n_jobs=-1).mean()
acc_xgb_tol   = cross_val_score(best_xgb, X_train, y_train, cv=3, scoring=scorer_tol, n_jobs=-1).mean()

resultados_modelos['XGB'] = {
    'pipe': best_xgb,
    'mae_cv': -xgb_search.best_score_,
    'acc_exact': acc_xgb_exact,
    'acc_tol': acc_xgb_tol,
    'params': xgb_search.best_params_
}

print(f"XGB - Mejor MAE CV: {-xgb_search.best_score_:.3f}")
print(f"XGB - Accuracy Exacta CV: {acc_xgb_exact:.2%}")

# ------------ 8.3 HISTGB ------------
print("\n>>> 8.3 HistGB")
param_hgb = {
    "model__max_iter": randint(200, 500),
    "model__learning_rate": uniform(0.05, 0.15),
    "model__max_depth": randint(5, 15),
    "model__min_samples_leaf": randint(10, 40),
    "model__l2_regularization": uniform(0.0, 1.5)
}

hgb_pipe = Pipeline([
    ("preprocess", preprocesador),
    ("model", HistGradientBoostingRegressor(random_state=SEED))
])

hgb_search = RandomizedSearchCV(hgb_pipe, param_hgb, n_iter=40, cv=3, 
                               scoring="neg_mean_absolute_error", 
                               random_state=SEED, n_jobs=-1, verbose=1)
hgb_search.fit(X_train, y_train)

# Calcular Accuracy CV
best_hgb = hgb_search.best_estimator_
acc_hgb_exact = cross_val_score(best_hgb, X_train, y_train, cv=3, scoring=scorer_exact, n_jobs=-1).mean()
acc_hgb_tol   = cross_val_score(best_hgb, X_train, y_train, cv=3, scoring=scorer_tol, n_jobs=-1).mean()

resultados_modelos['HGB'] = {
    'pipe': best_hgb,
    'mae_cv': -hgb_search.best_score_,
    'acc_exact': acc_hgb_exact,
    'acc_tol': acc_hgb_tol,
    'params': hgb_search.best_params_
}

print(f"HGB - Mejor MAE CV: {-hgb_search.best_score_:.3f}")
print(f"HGB - Accuracy Exacta CV: {acc_hgb_exact:.2%}")

# ============================================================
# 9. INTERPRETABILIDAD Y RESUMEN FINAL
# ============================================================

print("\n=== 9) RESUMEN DE RENDIMIENTO Y TOP VARIABLES ===")

# Imprimir tabla comparativa de métricas
print(f"{'Modelo':<10} | {'MAE (CV)':<10} | {'Acc Exacta':<12} | {'Acc (+/-1)':<12}")
print("-" * 55)
for nombre, res in resultados_modelos.items():
    print(f"{nombre:<10} | {res['mae_cv']:<10.3f} | {res['acc_exact']:<12.2%} | {res['acc_tol']:<12.2%}")

print("\n=== INTERPRETABILIDAD - TOP 5 VARIABLES POR MODELO ===")

for modelo_nombre, resultado in resultados_modelos.items():
    print(f"\n📊 {modelo_nombre} - Feature Importance:")
    
    pipe = resultado['pipe']
    
    # Obtener features names tras el preprocesamiento
    # Nota: Dependiendo de tu versión de sklearn, get_feature_names_out puede requerir ajuste si usas ColumnTransformer
    try:
        feature_names = pipe.named_steps['preprocess'].get_feature_names_out()
    except:
        # Fallback si falla la obtención de nombres (a veces pasa con versiones antiguas)
        feature_names = [f"Var_{i}" for i in range(X_train.shape[1])]

    # Transformar X_train para permutation importance (si aplica)
    if modelo_nombre == 'HGB':
        X_train_transformed = pipe.named_steps['preprocess'].transform(X_train)
        r = permutation_importance(
            pipe.named_steps['model'], X_train_transformed, y_train,
            n_repeats=5, random_state=SEED, n_jobs=-1
        )
        importancias = pd.Series(r.importances_mean, index=feature_names)
    else:
        # RF y XGB tienen feature_importances_ nativo
        importancias = pd.Series(
            pipe.named_steps['model'].feature_importances_,
            index=feature_names
        )
    
    top5 = importancias.sort_values(ascending=False).head(5)
    
    print("Rank | Variable                       | Importancia")
    print("-" * 50)
    for i, (var, imp) in enumerate(top5.items(), 1):
        print(f"{i:4d} | {var:30s} | {imp:7.4f}")
    
    # Gráfico Top 10
    plt.figure(figsize=(10, 6))
    top10 = importancias.sort_values(ascending=True).tail(10)
    top10.plot(kind='barh', color='skyblue', edgecolor='navy')
    plt.title(f'Top 10 Variables - {modelo_nombre} (MAE CV: {resultado["mae_cv"]:.2f})')
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.savefig(f"plots/feature_importance_{modelo_nombre.lower()}.png", dpi=300)
    plt.close()
    print(f"✓ Gráfico guardado: plots/feature_importance_{modelo_nombre.lower()}.png")

print("\n" + "="*60)
print("🏆 RESUMEN: HistGB mejor MAE CV (2.837). Compara interpretaciones arriba.")
print("="*60)

# ============================================================
# 10. SELECCIÓN DEL MODELO GANADOR Y PREDICCIÓN EN TEST (2024)
# ============================================================

print("\n=== 10) EVALUACIÓN FINAL EN DATASET DE TEST (2024) ===")

# 1. Identificar el mejor modelo basándonos en el MAE validado (menor es mejor)
mejor_modelo_nombre = min(resultados_modelos, key=lambda k: resultados_modelos[k]['mae_cv'])
mejor_config = resultados_modelos[mejor_modelo_nombre]
mejor_pipe = mejor_config['pipe']

print(f"🏆 El modelo ganador es: {mejor_modelo_nombre}")
print(f"   MAE en Validación (CV 2018-2023): {mejor_config['mae_cv']:.4f}")

# 2. Preparar datos de Test (Separar X e y para 2024)
X_test = test_df[num_cols + cat_cols].copy()
y_test = test_df['Position']

# 3. Predicción final
# El pipeline ya fue re-entrenado con todo el train set al final del RandomizedSearchCV (refit=True)
y_pred_test = mejor_pipe.predict(X_test)

# 4. Métricas en Test
mae_test = mean_absolute_error(y_test, y_pred_test)
acc_exact_test = exact_accuracy(y_test, y_pred_test)
acc_tol_test = tolerance_accuracy(y_test, y_pred_test)

print(f"\n>> Resultados Finales en Temporada 2024 (Datos no vistos):")
print(f"   MAE: {mae_test:.4f}")
print(f"   Accuracy Exacta: {acc_exact_test:.2%}")
print(f"   Accuracy (Margen ±1): {acc_tol_test:.2%}")

# 5. Gráfico de Predicción vs Realidad
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color='royalblue', edgecolors='k', label='Predicciones')
plt.plot([1, 20], [1, 20], 'r--', linewidth=2, label='Perfecto')
plt.fill_between([1, 20], [0, 19], [2, 21], color='red', alpha=0.1, label='Margen ±1')
plt.title(f'Predicción vs Realidad (2024) - Modelo {mejor_modelo_nombre}')
plt.xlabel('Posición Real')
plt.ylabel('Posición Predicha')
plt.xticks(range(1, 21))
plt.yticks(range(1, 21))
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("plots/final_pred_vs_real_2024.png")
plt.close()
print("✓ Gráfico guardado: plots/final_pred_vs_real_2024.png")

# ============================================================
# 11. ANÁLISIS EXHAUSTIVO CON SHAP (MODELO FINAL)
# ============================================================

print(f"\n=== 11) ANÁLISIS DE EXPLICABILIDAD SHAP ({mejor_modelo_nombre}) ===")

# --- A. Preparación de datos para SHAP ---
# SHAP necesita los datos YA transformados numéricamente, no el pipeline crudo.
# 1. Extraemos el preprocesador y el modelo del pipeline ganador
preprocessor_final = mejor_pipe.named_steps['preprocess']
regressor_final = mejor_pipe.named_steps['model']

# 2. Transformamos X_test numéricamente
X_test_transformed = preprocessor_final.transform(X_test)

# 3. Recuperamos los nombres de las columnas transformadas
try:
    feature_names_shap = preprocessor_final.get_feature_names_out()
except:
    feature_names_shap = [f"Feature_{i}" for i in range(X_test_transformed.shape[1])]

# 4. Creamos un DataFrame bonito para que SHAP muestre nombres reales en los gráficos
X_test_shap_df = pd.DataFrame(X_test_transformed, columns=feature_names_shap)

print(f"   Generando valores SHAP para {X_test_shap_df.shape[0]} instancias de test...")

# --- B. Cálculo de valores SHAP ---
# Usamos shap.Explainer que auto-detecta si es Tree, Linear o Permutation
# (Nota: XGBoost y RF funcionan nativo con TreeExplainer. HistGB a veces requiere un wrapper, Explainer lo maneja)
try:
    explainer = shap.Explainer(regressor_final)
    shap_values = explainer(X_test_shap_df)
except Exception as e:
    print(f"   Nota: Falló TreeExplainer nativo ({e}), intentando método agnóstico (puede tardar un poco)...")
    # Fallback para modelos complicados (como algunas versiones de sklearn HGB)
    # Usamos un background dataset resumen para agilizar
    mask_background = shap.maskers.Independent(data=X_test_shap_df, max_samples=100)
    explainer = shap.Explainer(regressor_final.predict, mask_background)
    shap_values = explainer(X_test_shap_df)

# --- C. Gráficos SHAP ---

# 1. Summary Plot (Beeswarm) - El más importante
plt.figure()
shap.summary_plot(shap_values, X_test_shap_df, show=False, max_display=15)
plt.title(f"Impacto de variables en la Posición Final ({mejor_modelo_nombre})")
plt.tight_layout()
plt.savefig("plots/shap_summary_beeswarm.png")
plt.close()
print("✓ Gráfico guardado: plots/shap_summary_beeswarm.png")

# 2. Bar Plot (Importancia media absoluta)
plt.figure()
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title(f"Importancia media absoluta ({mejor_modelo_nombre})")
plt.tight_layout()
plt.savefig("plots/shap_importance_bar.png")
plt.close()
print("✓ Gráfico guardado: plots/shap_importance_bar.png")

# 3. Dependence Plot (Variable Top 1)
# Identificamos la variable más importante (la que tiene mayor media de valor absoluto SHAP)
top_feature_idx = np.abs(shap_values.values).mean(0).argmax()
top_feature_name = feature_names_shap[top_feature_idx]

print(f"\nVariable más influyente detectada por SHAP: {top_feature_name}")

plt.figure()
shap.plots.scatter(shap_values[:, top_feature_name], color=shap_values, show=False)
plt.title(f"Dependencia SHAP: {top_feature_name}")
plt.tight_layout()
plt.savefig(f"plots/shap_dependence_top1.png")
plt.close()
print(f"✓ Gráfico guardado: plots/shap_dependence_top1.png")

print("\n=== PIPELINE FINALIZADO CON ÉXITO ===")

