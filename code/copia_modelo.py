import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import shap

# Librerías de Preprocesado y Validación
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import randint, uniform
import warnings
import os

# ============================================================
# 0. CONFIGURACIÓN GENERAL
# ============================================================

SEED = 6967
np.random.seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs("plots", exist_ok=True)

print("\n=== INICIO PIPELINE: MODELO EXPLICATIVO F1 POSITION ===\n")

# ============================================================
# 1. CARGA Y LIMPIEZA INICIAL
# ============================================================

datos = pd.read_csv("data/f1_pitstops_2018_2024.csv", sep=",")

# Gestión de DNFs y Vueltas
vueltas_max = datos.groupby(['Season', 'Circuit'])['Laps'].max().rename('Max_Laps').reset_index()
datos = datos.merge(vueltas_max, on=['Season', 'Circuit'], how='left')
mask_finished = datos['Laps'] >= (datos['Max_Laps'] - 1)
datos = datos[mask_finished].copy()
datos.drop(columns=['Max_Laps', "Location", "Country", "Date", "Round", 
                    "Race Name", "Time_of_race", "Total Pit Stops", "Fast Lap Attempts"], inplace=True)

# Tratamiento de tipos
datos['Pit_Lap'] = pd.to_numeric(datos['Pit_Lap'].replace('', np.nan), errors='coerce')
datos['Tire Compound'] = datos['Tire Compound'].astype(str).replace('nan', 'Unknown')
datos['Pit_Time'] = datos['Pit_Time'].astype(str)
mask_final_stint = datos['Pit_Time'] == "Final Stint"
datos.loc[mask_final_stint, 'Pit_Lap'] = datos.loc[mask_final_stint, 'Pit_Lap'].fillna(-1)

# ============================================================
# 2. DEFINICIÓN DE VARIABLES Y SPLIT TEMPORAL
# ============================================================

num_cols = [
    'TotalPitStops', 'AvgPitStopTime', 'Air_Temp_C',
    'Track_Temp_C', 'Humidity_%', 'Wind_Speed_KMH',
    'Pit_Lap', 'Driver Aggression Score', 'Tire Usage Aggression'
]
cat_cols = ['Driver', 'Constructor', 'Tire Compound', 'Circuit']

# SPLIT TEMPORAL RIGUROSO (Feedback Punto 2)
train_df = datos[datos['Season'].between(2018, 2023)].copy()
test_df  = datos[datos['Season'] == 2024].copy()

X_train = train_df[num_cols + cat_cols].copy()
y_train = train_df['Position']
X_test = test_df[num_cols + cat_cols].copy()
y_test = test_df['Position']

print(f"Train (2018-2023): {X_train.shape[0]} filas")
print(f"Test (2024): {X_test.shape[0]} filas")

# ============================================================
# 3. ANÁLISIS DE MULTICOLINEALIDAD (Feedback Punto 4)
# ============================================================
print("\n=== 3. ANÁLISIS DE CORRELACIÓN (MULTICOLINEALIDAD) ===")
# Solo variables numéricas para el heatmap
corr_matrix = train_df[num_cols].corr()

plt.figure(figsize=(10, 8))
# Usamos un mapa de calor divergente para ver correlaciones fuertes pos/neg
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matriz de Correlación - Variables Numéricas")
plt.tight_layout()
plt.savefig("plots/correlation_matrix.png")
plt.close()
print("✓ Gráfico de correlación guardado: plots/correlation_matrix.png")
print(">> Nota: Revisa este gráfico para la memoria. Si ves valores > 0.8 (ej: Track_Temp vs Air_Temp), coméntalo en el informe.")

# ============================================================
# 4. SELECCIÓN DE MEJOR IMPUTADOR/ESCALADOR (KNN)
# ============================================================
print("\n=== 4. OPTIMIZACIÓN DE PREPROCESADO (KNN) ===")

scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]
imputers = [SimpleImputer(strategy="mean"), SimpleImputer(strategy="median")]
mejor_config_pre = {'scaler': None, 'imputer': None, 'score': np.inf}

# Definimos TimeSeriesSplit para la validación interna también (Coherencia total)
tscv_pre = TimeSeriesSplit(n_splits=3)

for scaler in scalers:
    for imputer in imputers:
        num_trans = Pipeline(steps=[("imputer", imputer), ("scaler", scaler)])
        cat_trans = Pipeline(steps=[("enc", OneHotEncoder(handle_unknown="ignore"))])
        
        pre = ColumnTransformer([("num", num_trans, num_cols), ("cat", cat_trans, cat_cols)])
        pipe = Pipeline([("pre", pre), ("model", KNeighborsRegressor(n_neighbors=5))])
        
        # Usamos tscv aquí también para evitar data leakage en la selección del preprocesador
        scores = cross_val_score(pipe, X_train, y_train, cv=tscv_pre, scoring="neg_mean_absolute_error", n_jobs=-1)
        mae = -scores.mean()
        
        if mae < mejor_config_pre['score']:
            mejor_config_pre = {'scaler': scaler, 'imputer': imputer, 'score': mae}

print(f"Mejor configuración: {mejor_config_pre['imputer']} + {mejor_config_pre['scaler']}")

# Construcción del Preprocesador Final
numerical_final = Pipeline(steps=[("imputer", mejor_config_pre['imputer']), ("scaler", mejor_config_pre['scaler'])])
categorical_final = Pipeline(steps=[("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocesador = ColumnTransformer([("num", numerical_final, num_cols), ("cat", categorical_final, cat_cols)], sparse_threshold=0)

# ============================================================
# 5. ENTRENAMIENTO Y AJUSTE DE HIPERPARÁMETROS
# ============================================================
print("\n=== 5. AJUSTE DE MODELOS CON VALIDACIÓN TEMPORAL (TimeSeriesSplit) ===")

# Métricas personalizadas
def exact_acc(y_t, y_p): return np.mean(y_t == np.round(y_p))
def tol_acc(y_t, y_p): return np.mean(np.abs(y_t - np.round(y_p)) <= 1)
scorer_exact = make_scorer(exact_acc)
scorer_tol = make_scorer(tol_acc)

resultados = {}

# Definimos la estrategia de validación cruzada temporal para TODOS los modelos
# Esto responde directamente al Feedback punto 2: "Estrategia de validación"
tscv = TimeSeriesSplit(n_splits=4) 

# --- A. Random Forest ---
print(">> Entrenando Random Forest...")
rf_pipe = Pipeline([("pre", preprocesador), ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
rf_params = {
    "model__n_estimators": randint(100, 300),
    "model__max_depth": randint(5, 20),
    "model__min_samples_leaf": randint(2, 10)
}
rf_search = RandomizedSearchCV(rf_pipe, rf_params, n_iter=20, cv=tscv, # USO DE TSCV
                               scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
rf_search.fit(X_train, y_train)
resultados['RF'] = rf_search

# --- B. XGBoost ---
print(">> Entrenando XGBoost...")
xgb_pipe = Pipeline([("pre", preprocesador), ("model", XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist"))])
xgb_params = {
    "model__n_estimators": randint(100, 400),
    "model__learning_rate": uniform(0.01, 0.2),
    "model__max_depth": randint(3, 8),
    "model__subsample": uniform(0.6, 0.4)
}
xgb_search = RandomizedSearchCV(xgb_pipe, xgb_params, n_iter=30, cv=tscv, # USO DE TSCV
                                scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
xgb_search.fit(X_train, y_train)
resultados['XGB'] = xgb_search

# --- C. HistGradientBoosting ---
print(">> Entrenando HistGradientBoosting...")
hgb_pipe = Pipeline([("pre", preprocesador), ("model", HistGradientBoostingRegressor(random_state=SEED))])
hgb_params = {
    "model__max_iter": randint(100, 300),
    "model__learning_rate": uniform(0.05, 0.2),
    "model__max_depth": randint(5, 15),
    "model__l2_regularization": uniform(0.0, 1.0)
}
hgb_search = RandomizedSearchCV(hgb_pipe, hgb_params, n_iter=30, cv=tscv, # USO DE TSCV
                                scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
hgb_search.fit(X_train, y_train)
resultados['HGB'] = hgb_search

# ============================================================
# 6. ANÁLISIS DE IMPORTANCIA Y SELECCIÓN DE MODELO
# ============================================================

print("\n=== 6. COMPARATIVA E INTERPRETABILIDAD POR MODELO ===")

# --- 6.1 Feature Importance Individual (Añadido de tu versión anterior) ---
print("\n--- Análisis de Variables Top 5 por Modelo ---")
for nombre, search in resultados.items():
    print(f"\n📊 {nombre} - Feature Importance:")
    
    # Extraemos el mejor pipeline del RandomSearch
    best_pipe = search.best_estimator_
    model_step = best_pipe.named_steps['model']
    pre_step = best_pipe.named_steps['pre']
    
    # Obtener feature names del preprocesador
    try:
        feature_names = pre_step.get_feature_names_out()
    except:
        feature_names = [f"Var_{i}" for i in range(X_train.shape[1])]

    # Calcular importancia
    if nombre == 'HGB':
        # HGB necesita permutation importance porque no tiene feature_importances_ directo robusto
        # Transformamos X_train para pasarlo al modelo
        X_train_trans = pre_step.transform(X_train)
        r = permutation_importance(model_step, X_train_trans, y_train,
                                   n_repeats=5, random_state=SEED, n_jobs=-1)
        importancias = pd.Series(r.importances_mean, index=feature_names)
    else:
        # RF y XGB tienen atributo nativo
        importancias = pd.Series(model_step.feature_importances_, index=feature_names)
    
    # Imprimir Top 5
    top5 = importancias.sort_values(ascending=False).head(5)
    print("Rank | Variable                       | Importancia")
    print("-" * 50)
    for i, (var, imp) in enumerate(top5.items(), 1):
        print(f"{i:4d} | {var:30s} | {imp:7.4f}")
    
    # Guardar Gráfico Top 10
    plt.figure(figsize=(10, 6))
    top10 = importancias.sort_values(ascending=True).tail(10)
    top10.plot(kind='barh', color='skyblue', edgecolor='navy')
    plt.title(f'Top 10 Variables - {nombre} (MAE CV: {-search.best_score_:.2f})')
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.savefig(f"plots/feature_importance_{nombre.lower()}.png", dpi=300)
    plt.close()
    print(f"✓ Gráfico guardado: plots/feature_importance_{nombre.lower()}.png")

# --- 6.2 Tabla Resumen de Métricas ---
print("\n--- Tabla Comparativa de Rendimiento (Validación Temporal) ---")
print(f"{'Modelo':<10} | {'MAE (CV)':<20} | {'Mejores Params'}")
print("-" * 80)

best_model_name = ""
best_mae = np.inf
best_pipe = None

for name, search in resultados.items():
    mae = -search.best_score_
    print(f"{name:<10} | {mae:<20.4f} | {str(search.best_params_)[:40]}...")
    
    if mae < best_mae:
        best_mae = mae
        best_model_name = name
        best_pipe = search.best_estimator_

print("-" * 80)
print(f"🏆 MODELO SELECCIONADO: {best_model_name} (Menor Error en Validación Temporal)")

# ============================================================
# 7. EVALUACIÓN FINAL EN TEST 2024
# ============================================================
print(f"\n=== 7. VALIDACIÓN FINAL: TEMPORADA 2024 ({best_model_name}) ===")

# Predicción
y_pred = best_pipe.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred)
acc_exact = exact_acc(y_test, y_pred)
acc_tol = tol_acc(y_test, y_pred)

print(f"MAE Test 2024: {mae_test:.4f}")
print(f"Exactitud (Puesto Exacto): {acc_exact:.2%}")
print(f"Exactitud (Margen ±1): {acc_tol:.2%}")

# Gráfico Predicción vs Realidad
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicción')
plt.plot([1, 20], [1, 20], 'r--', label='Ideal')
plt.fill_between([1, 20], [0, 19], [2, 21], color='red', alpha=0.1)
plt.xlabel("Posición Real (2024)")
plt.ylabel("Posición Predicha")
plt.title(f"Desempeño en Test 2024 - {best_model_name}")
plt.legend()
plt.tight_layout()
plt.savefig("plots/pred_vs_real_2024.png")
plt.close()
print("✓ Gráfico guardado: plots/pred_vs_real_2024.png")

# ============================================================
# 8. ANÁLISIS SHAP PARA INTERPRETABILIDAD FINAL
# ============================================================
print("\n=== 8. ANÁLISIS SHAP AVANZADO (MODELO GANADOR) ===")

# 1. Preparar datos transformados para SHAP
prep = best_pipe.named_steps['pre']
model = best_pipe.named_steps['model']

# Transformamos los datos de test manualmente
X_test_trans = prep.transform(X_test)

# 2. Obtener nombres de features (para que el gráfico se entienda)
try:
    feat_names = prep.get_feature_names_out()
except:
    feat_names = [f"F{i}" for i in range(X_test_trans.shape[1])]

# Convertimos a DataFrame para SHAP
X_test_df = pd.DataFrame(X_test_trans, columns=feat_names)

print(f"   Calculando SHAP para {len(X_test_df)} muestras...")

# 3. Calcular SHAP con manejo de errores de Aditividad
try:
    # Intento 1: TreeExplainer estándar (Rápido)
    # check_additivity=False soluciona el error de precisión en modelos Hist/XGB
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_df, check_additivity=False)

except Exception as e:
    print(f"   >> Nota: TreeExplainer falló ({e}). Usando método agnóstico...")
    # Intento 2: Explainer genérico (Fallback)
    masker = shap.maskers.Independent(data=X_test_df, max_samples=100)
    explainer = shap.Explainer(model.predict, masker=masker)
    shap_values = explainer(X_test_df)

# 4. Generar Gráficos
print("   Generando gráficos SHAP...")

# Gráfico Beeswarm (Resumen Global)
plt.figure()
shap.summary_plot(shap_values, X_test_df, show=False, max_display=12)
plt.title(f"Drivers del Modelo ({best_model_name}) - SHAP Beeswarm")
plt.tight_layout()
plt.savefig("plots/shap_beeswarm.png")
plt.close()
print("✓ Guardado: plots/shap_beeswarm.png")

# Gráfico de Barras (Importancia Absoluta)
plt.figure()
shap.plots.bar(shap_values, max_display=12, show=False)
plt.title(f"Importancia de Variables ({best_model_name}) - SHAP Bar")
plt.tight_layout()
plt.savefig("plots/shap_bar.png")
plt.close()
print("✓ Guardado: plots/shap_bar.png")

# Gráfico de Dependencia (Top 1 Variable)
try:
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_feature_idx = np.argmax(mean_abs_shap)
    top_feature_name = feat_names[top_feature_idx]
    
    plt.figure()
    shap.plots.scatter(shap_values[:, top_feature_name], color=shap_values, show=False)
    plt.title(f"Dependencia SHAP: {top_feature_name}")
    plt.tight_layout()
    plt.savefig("plots/shap_dependence_top1.png")
    plt.close()
    print(f"✓ Guardado: plots/shap_dependence_top1.png (Variable: {top_feature_name})")
except Exception as e:
    print(f"   (No se pudo generar el gráfico de dependencia: {e})")

print("\n=== PROCESO COMPLETADO ===")