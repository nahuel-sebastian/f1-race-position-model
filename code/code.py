# MODELO EXPLICATIVO DE PREDICCIÓN DE POSICIÓN EN CARRERA DE F1 (2018-2024)

# Grupo 9 de practicas
# Javier Rosales Lozano
# Manuel Roldán Matea
# Nahuel Sebastián Vargas
# Alejandro Rodríguez García

# Importacion de librerias
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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import randint, uniform
import warnings
import os

# Configuraciones iniciales
warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs("plots", exist_ok=True)
SEED = 6967
np.random.seed(SEED)

# CARGA Y LIMPIEZA
print("CARGA Y LIMPIEZA")
try:
    datos = pd.read_csv("data/f1_pitstops_2018_2024.csv", sep=",")
except:
    print(" Archivo no encontrado. Crea 'data/f1_pitstops_2018_2024.csv'")
    exit()
print(f"Tamaño inicial: {datos.shape}")

# Aqui se eliminan las columnas que no aportan valor al modelo y no tienen un peso predictivo dentro del modelo
# Se filtran los DNF de los pilosos para que el modelo solo  trabaje como carreras de pilotos en las que ha finalizado
vueltas_max = datos.groupby(['Season', 'Circuit'])['Laps'].max().rename('Max_Laps').reset_index()
datos = datos.merge(vueltas_max, on=['Season', 'Circuit'], how='left')
mask_finished = datos['Laps'] >= (datos['Max_Laps'] - 1)
datos = datos[mask_finished].copy()
datos.drop(columns=['Max_Laps'], inplace=True)
datos.drop(columns=["Location", "Country", "Date", "Round", "Race Name", "Time_of_race", 
                   "Total Pit Stops", "Fast Lap Attempts"], inplace=True, errors='ignore')

# Normalizar tipos de las variables como Pit_Lap, Tire Compound y Pit_Time
datos['Pit_Lap'] = pd.to_numeric(datos['Pit_Lap'].replace('', np.nan), errors='coerce')
datos['Tire Compound'] = datos['Tire Compound'].astype(str).replace('nan', 'Unknown')
datos['Pit_Time'] = datos['Pit_Time'].astype(str)
mask_final_stint = datos['Pit_Time'] == "Final Stint"
datos.loc[mask_final_stint, 'Pit_Lap'] = datos.loc[mask_final_stint, 'Pit_Lap'].fillna(-1)

print(f"Dataset limpio: {datos.shape}")

# División en train/test temporal (2018-2023 / 2024) para que  el modelo no tenga data leakage
num_cols = ['TotalPitStops', 'AvgPitStopTime', 'Air_Temp_C', 'Track_Temp_C', 'Humidity_%', 
           'Wind_Speed_KMH', 'Pit_Lap', 'Driver Aggression Score', 'Tire Usage Aggression']
cat_cols = ['Driver', 'Constructor', 'Tire Compound', 'Circuit']

train_df = datos[datos['Season'].between(2018, 2023)].copy()
test_df = datos[datos['Season'] == 2024].copy()
X_train = train_df[num_cols + cat_cols].copy()
y_train = train_df['Position']
X_test = test_df[num_cols + cat_cols].copy()
y_test = test_df['Position']
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Análisis de la colinealidad de las variables restantes del dataset
print("\nMULTICOLINEALIDAD")
corr_matrix = train_df[num_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.savefig("plots/correlation_matrix.png", dpi=300)
plt.close()

high_corr_pairs = [(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]) 
                  for i in range(len(corr_matrix.columns)) 
                  for j in range(i+1, len(corr_matrix.columns)) 
                  if abs(corr_matrix.iloc[i, j]) > 0.8]
for var1, var2, corr in high_corr_pairs:
    print(f" {var1} con {var2}: {corr:.3f}")
print("correlation_matrix.png")

# PREPROCESADOR OPTIMIZADO
# Se prueba distintas combinaciones de escaladores e imputadores para encontrar la mejor configuración con KNN
print("\nPREPROCESADOR")
scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]
imputers = [SimpleImputer(strategy="mean"), SimpleImputer(strategy="median")]
mejor_config_pre = {'scaler': None, 'imputer': None, 'score': np.inf}

tscv_pre = TimeSeriesSplit(n_splits=3)
for scaler in scalers:
    for imputer in imputers:
        num_trans = Pipeline([("imputer", imputer), ("scaler", scaler)])
        cat_trans = Pipeline([("enc", OneHotEncoder(handle_unknown="ignore"))])
        pre = ColumnTransformer([("num", num_trans, num_cols), ("cat", cat_trans, cat_cols)])
        pipe = Pipeline([("pre", pre), ("model", KNeighborsRegressor(n_neighbors=5))])
        scores = cross_val_score(pipe, X_train, y_train, cv=tscv_pre, 
                               scoring="neg_mean_absolute_error", n_jobs=-1)
        mae = -scores.mean()
        if mae < mejor_config_pre['score']:
            mejor_config_pre = {'scaler': scaler, 'imputer': imputer, 'score': mae}

#Resultados del preprocesador

print(f"Mejor Preprocesador: Imputer={type(mejor_config_pre['imputer']).__name__}, "
        f"Scaler={type(mejor_config_pre['scaler']).__name__}, MAE={mejor_config_pre['score']:.4f}")

mejor_scaler = mejor_config_pre['scaler']
mejor_imputer = mejor_config_pre['imputer']
numerical_final = Pipeline([("imputer", mejor_imputer), ("scaler", mejor_scaler)])
categorical_final = Pipeline([("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocesador = ColumnTransformer([("num", numerical_final, num_cols), 
                                  ("cat", categorical_final, cat_cols)], sparse_threshold=0)

# ENTRENAMIENTO COMPLETO 
print("\nENTRENAMIENTO DE MODELOS")
def exact_acc(y_t, y_p): return np.mean(y_t == np.round(y_p))
def tol_acc(y_t, y_p): return np.mean(np.abs(y_t - np.round(y_p)) <= 1)
scorer_exact = make_scorer(exact_acc)
scorer_tol = make_scorer(tol_acc)

resultados_modelos = {}
tscv = TimeSeriesSplit(n_splits=4)

# Entranamiento y optimización de hiperparámetros de random forest 
print("Random Forest")
rf_pipe = Pipeline([("pre", preprocesador), ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
rf_params = {"model__n_estimators": randint(200, 600), "model__max_depth": randint(10, 25),
            "model__min_samples_leaf": randint(1, 8), "model__max_features": ["sqrt", "log2"]}
rf_search = RandomizedSearchCV(rf_pipe, rf_params, n_iter=25, cv=3, 
                              scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
acc_rf_exact = cross_val_score(best_rf, X_train, y_train, cv=3, scoring=scorer_exact, n_jobs=-1).mean()
resultados_modelos['RF'] = {'pipe': best_rf, 'mae_cv': -rf_search.best_score_, 'acc_exact': acc_rf_exact}

# Entranamiento y optimización de hiperparámetros de XgBoost
print("XGBoost")
xgb_pipe = Pipeline([("pre", preprocesador), ("model", XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist"))])
xgb_params = {"model__n_estimators": randint(200, 500), "model__learning_rate": uniform(0.01, 0.2),
             "model__max_depth": randint(3, 10), "model__subsample": uniform(0.7, 0.3)}
xgb_search = RandomizedSearchCV(xgb_pipe, xgb_params, n_iter=30, cv=3, 
                               scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
acc_xgb_exact = cross_val_score(best_xgb, X_train, y_train, cv=3, scoring=scorer_exact, n_jobs=-1).mean()
resultados_modelos['XGB'] = {'pipe': best_xgb, 'mae_cv': -xgb_search.best_score_, 'acc_exact': acc_xgb_exact}

# Entranamiento y optimización de hiperparámetros de HistGradientBoosting
print("HistGradientBoosting")
hgb_pipe = Pipeline([("pre", preprocesador), ("model", HistGradientBoostingRegressor(random_state=SEED))])
hgb_params = {"model__max_iter": randint(200, 400), "model__learning_rate": uniform(0.05, 0.15),
             "model__max_depth": randint(5, 12)}
hgb_search = RandomizedSearchCV(hgb_pipe, hgb_params, n_iter=30, cv=3, 
                               scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
hgb_search.fit(X_train, y_train)
best_hgb = hgb_search.best_estimator_
acc_hgb_exact = cross_val_score(best_hgb, X_train, y_train, cv=3, scoring=scorer_exact, n_jobs=-1).mean()
resultados_modelos['HGB'] = {'pipe': best_hgb, 'mae_cv': -hgb_search.best_score_, 'acc_exact': acc_hgb_exact}

# Tablas comparativas de resultados de cada uno de los modelos  
print("\nTABLA COMPARATIVA")
print(f"{'Modelo':<10} {'MAE CV':<10} {'Acc Exacta':<12}")
print("-" * 40)
for name, res in resultados_modelos.items():
    print(f"{name:<10} {res['mae_cv']:<10.3f} {res['acc_exact']:<12.2%}")

mejor_modelo_nombre = min(resultados_modelos, key=lambda k: resultados_modelos[k]['mae_cv'])
mejor_pipe = resultados_modelos[mejor_modelo_nombre]['pipe']
print(f"\n GANADOR: {mejor_modelo_nombre}")

# Tabla que muestra la feature importance del mejor modelo seleccionado anteoriormente a partir del MAE y su accuracy
print(f"\nFEATURE IMPORTANCE {mejor_modelo_nombre} ")
pre_step = mejor_pipe.named_steps['pre']
model_step = mejor_pipe.named_steps['model']
try:
    feat_names = pre_step.get_feature_names_out()
except:
    feat_names = [f'Var{i}' for i in range(100)]

X_train_trans = pre_step.transform(X_train[:2000])  # Submalla para velocidad
if mejor_modelo_nombre == 'HGB':
    r = permutation_importance(model_step, X_train_trans, y_train[:2000], n_repeats=3, random_state=SEED)
    importancias = pd.Series(r.importances_mean, index=feat_names[:len(r.importances_mean)])
else:
    importancias = pd.Series(model_step.feature_importances_, index=feat_names[:len(model_step.feature_importances_)])

# Plot de las 10 variables más importantes del modelo
top10 = importancias.nlargest(10)
for i, (var, imp) in enumerate(top10.items(), 1):
    print(f"{i:2d}. {str(var)[:40]} {imp:.4f}")
top10.plot(kind='barh', color='skyblue')
plt.title(f"Top 10 Variables - {mejor_modelo_nombre}")
plt.xlabel('Importancia')
plt.tight_layout()
plt.savefig(f"plots/feature_importance_{mejor_modelo_nombre.lower()}.png", dpi=300)
plt.close()

# Test final con el mejor modelo para 2024 y gráfico de predicción vs realidad con margen visual ±1
print("\nTEST 2024 CON MARGEN VISUAL ±1 ")
y_pred = mejor_pipe.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred)
print(f"MAE Test: {mae_test:.4f}")
print(f"Acc Exacta: {exact_acc(y_test, y_pred):.1%}")
print(f"Acc ±1: {tol_acc(y_test, y_pred):.1%}")
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue', s=50, edgecolors='darkblue', linewidth=0.5)
plt.plot([1, 20], [1, 20], 'r--', lw=3, label='Perfecto (pred = real)', alpha=0.9)

#  Lineas de margen ±1
plt.plot([1, 20], [2, 21], 'orange', lw=2, linestyle=':', alpha=0.8, label='Margen +1')
plt.plot([1, 20], [0, 19], 'orange', lw=2, linestyle=':', alpha=0.8, label='Margen -1')

# Zona de la tolerancia de +- posición implementada
plt.fill_between([1, 20], [0, 19], [2, 21], color='orange', alpha=0.15, label='Zona ±1 posición')
plt.xlabel('Posición Real 2024', fontsize=12, fontweight='bold')
plt.ylabel('Posición Predicha', fontsize=12, fontweight='bold')
plt.title(f'Predicción vs Realidad 2024 - {mejor_modelo_nombre}\nMAE: {mae_test:.2f}', fontsize=14, fontweight='bold')
plt.xlim(0.5, 20.5)
plt.ylim(0.5, 20.5)
plt.xticks(range(1, 21))
plt.yticks(range(1, 21))
plt.grid(True, linestyle=':', alpha=0.3)
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('plots/pred_vs_real_2024.png', dpi=300, bbox_inches='tight')
plt.close()
print("pred_vs_real_2024.png (con margen ±1 visual)")
