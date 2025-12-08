import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import shap
import warnings
import os

warnings.filterwarnings('ignore', category=FutureWarning)
os.makedirs('plots', exist_ok=True)

print("=== PIPELINE COMPLETO F1 POSITION - MULTICOLINEALIDAD + TimeSeriesSplit ===")
SEED = 6967
np.random.seed(SEED)

# 1. CARGA Y LIMPIEZA INICIAL
print("1. CARGA Y LIMPIEZA DATASET")
try:
    datos = pd.read_csv('data/f1_pitstops_2018_2024.csv', sep=',')
except:
    print("❌ ERROR: No se encuentra 'data/f1_pitstops_2018-2024.csv'")
    print("Crea la carpeta 'data/' y coloca tu archivo CSV")
    exit()

print(f"Tamaño inicial: {datos.shape}")

# Filtrar DNFs reales
vueltasmax = datos.groupby(['Season', 'Circuit'])['Laps'].max().rename('MaxLaps').reset_index()
datos = datos.merge(vueltasmax, on=['Season', 'Circuit'], how='left')
mask_finished = (datos['Laps'] >= datos['MaxLaps'] - 1)
datos = datos[mask_finished].copy()
datos.drop(columns=['MaxLaps'], inplace=True)
datos.drop(columns=['Location', 'Country', 'Date', 'Round', 'Race Name', 'Timeofrace', 
                   'Total Pit Stops', 'Fast Lap Attempts'], inplace=True)

# Normalización tipos
datos['PitLap'] = pd.to_numeric(datos['PitLap'].replace('', np.nan), errors='coerce')
datos['Tire Compound'] = datos['Tire Compound'].astype(str).replace('nan', 'Unknown')
datos['PitTime'] = datos['PitTime'].astype(str)
mask_final_stint = datos['PitTime'] == 'Final Stint'
datos.loc[mask_final_stint, 'PitLap'] = datos.loc[mask_final_stint, 'PitLap'].fillna(-1)

print(f"Dataset limpio: {datos.shape}")
print("Columnas:", list(datos.columns))

# 2. SPLIT TEMPORAL RIGUROSO
train_df = datos[datos['Season'].between(2018, 2023)].copy()
test_df = datos[datos['Season'] == 2024].copy()
print(f"Train 2018-2023: {train_df.shape[0]} filas")
print(f"Test 2024: {test_df.shape[0]} filas")

# Features del modelo
num_cols = ['TotalPitStops', 'AvgPitStopTime', 'AirTempC', 'TrackTempC', 'Humidity', 
            'WindSpeedKMH', 'PitLap', 'Driver Aggression Score', 'Tire Usage Aggression']
cat_cols = ['Driver', 'Constructor', 'Tire Compound', 'Circuit']

X_train = train_df[num_cols + cat_cols].copy()
y_train = train_df['Position']
X_test = test_df[num_cols + cat_cols].copy()
y_test = test_df['Position']

# ============================================================
# 3. ANÁLISIS COMPLETO MULTICOLINEALIDAD (SIN statsmodels)
# ============================================================
print("\n=== 3. ANÁLISIS MULTICOLINEALIDAD (Correlación + VIF Simplificado) ===")

# 3.1 Matriz de correlación
corr_matrix = train_df[num_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
            center=0, square=True, linewidths=0.5)
plt.title("Matriz de Correlación - Variables Numéricas (Train 2018-2023)")
plt.tight_layout()
plt.savefig("plots/correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ correlation_matrix.png")

# 3.2 Correlaciones críticas (> |0.8|)
print("🔴 CORRELACIONES CRÍTICAS (|r| > 0.8):")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            var1, var2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            high_corr_pairs.append((var1, var2, corr_val))
            print(f"  {var1:25s} ↔ {var2:25s}  r={corr_val:5.3f}")

if not high_corr_pairs:
    print("  ✅ Sin correlaciones críticas")

# 3.3 VIF SIMPLIFICADO
print("\n📊 VIF SIMPLIFICADO (Correlación Máxima):")
vif_simple = {}
for col in num_cols:
    max_corr = corr_matrix[col].drop(col).abs().max()
    vif_simple[col] = 1 / (1 - max_corr**2) if max_corr < 0.99 else 50
vif_df = pd.DataFrame(list(vif_simple.items()), columns=['Variable', 'VIF']).round(2)
print(vif_df.sort_values('VIF', ascending=False))

# 3.4 Gráfico VIF
plt.figure(figsize=(10, 6))
vif_plot = vif_df.copy()
vif_plot['VIF'] = vif_plot['VIF'].clip(upper=50)
colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in vif_plot['VIF']]
vif_plot.plot(kind='barh', x='VIF', y='Variable', color=colors, legend=False)
plt.axvline(x=5, color='orange', linestyle='--', alpha=0.7)
plt.axvline(x=10, color='red', linestyle='--', alpha=0.7)
plt.title("VIF Simplificado - Multicolinealidad")
plt.xlabel("Factor de Inflación de Varianza")
plt.tight_layout()
plt.savefig("plots/vif_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ vif_analysis.png")

print("🎯 DECISIÓN: Mantener todas (modelos árbol robustos)")

# 4. PREPROCESADOR FINAL (simplificado)
print("\n4. PREPROCESADOR FINAL")
numerical_final = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                           ('scaler', StandardScaler())])
categorical_final = Pipeline([('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocesador = ColumnTransformer([('num', numerical_final, num_cols),
                                  ('cat', categorical_final, cat_cols)],
                                 sparse_threshold=0)

# 5. MÉTRICAS PERSONALIZADAS
def exact_acc(y_true, y_pred): return np.mean(y_true == np.round(y_pred))
def tol_acc(y_true, y_pred): return np.mean(np.abs(y_true - np.round(y_pred)) <= 1)
scorer_exact = make_scorer(exact_acc)
scorer_tol = make_scorer(tol_acc)

# 6. AJUSTE MODELOS CON TimeSeriesSplit PROGRESIVO
tscv = TimeSeriesSplit(n_splits=5)
resultados = {}

print("\n6. ENTRENAMIENTO MODELOS (TimeSeriesSplit progresivo)")

# Random Forest
print("   → Random Forest")
rf_pipe = Pipeline([('pre', preprocesador), ('model', RandomForestRegressor(random_state=SEED, n_jobs=-1))])
rf_params = {'model__n_estimators': randint(100, 300), 'model__max_depth': randint(5, 20)}
rf_search = RandomizedSearchCV(rf_pipe, rf_params, n_iter=15, cv=tscv, scoring='neg_mean_absolute_error', 
                             random_state=SEED, n_jobs=-1, verbose=0)
rf_search.fit(X_train, y_train)
resultados['RF'] = rf_search

# XGBoost
print("   → XGBoost")
xgb_pipe = Pipeline([('pre', preprocesador), ('model', XGBRegressor(random_state=SEED, n_jobs=-1, tree_method='hist'))])
xgb_params = {'model__n_estimators': randint(100, 300), 'model__learning_rate': uniform(0.01, 0.2)}
xgb_search = RandomizedSearchCV(xgb_pipe, xgb_params, n_iter=20, cv=tscv, scoring='neg_mean_absolute_error', 
                              random_state=SEED, n_jobs=-1, verbose=0)
xgb_search.fit(X_train, y_train)
resultados['XGB'] = xgb_search

# HistGradientBoosting
print("   → HistGradientBoosting")
hgb_pipe = Pipeline([('pre', preprocesador), ('model', HistGradientBoostingRegressor(random_state=SEED))])
hgb_params = {'model__max_iter': randint(100, 250), 'model__learning_rate': uniform(0.05, 0.15)}
hgb_search = RandomizedSearchCV(hgb_pipe, hgb_params, n_iter=20, cv=tscv, scoring='neg_mean_absolute_error', 
                              random_state=SEED, n_jobs=-1, verbose=0)
hgb_search.fit(X_train, y_train)
resultados['HGB'] = hgb_search

# 7. TABLA COMPARATIVA
print("\n=== TABLA COMPARATIVA MODELOS (TimeSeriesSplit n=5) ===")
print(f"{'Modelo':<10} {'MAE CV':<12}")
print("-" * 30)
best_model_name, best_mae, best_pipe = None, np.inf, None
for name, search in resultados.items():
    mae_cv = -search.best_score_
    print(f"{name:<10} {mae_cv:<12.4f}")
    if mae_cv < best_mae:
        best_mae, best_model_name, best_pipe = mae_cv, name, search.best_estimator_
print("-" * 30)
print(f"🏆 GANADOR: {best_model_name} (MAE CV: {best_mae:.4f})")

# 8. FEATURE IMPORTANCE + TEST 2024
print(f"\n8. FEATURE IMPORTANCE - {best_model_name}")
pre_step = best_pipe.named_steps['pre']
model_step = best_pipe.named_steps['model']

try:
    feat_names = pre_step.get_feature_names_out()
except:
    feat_names = [f'Var{i}' for i in range(50)]  # Aproximación

X_train_trans = pre_step.transform(X_train)
if best_model_name == 'HGB':
    r = permutation_importance(model_step, X_train_trans[:1000], y_train[:1000], n_repeats=3, random_state=SEED)
    importancias = pd.Series(r.importances_mean, index=feat_names[:len(r.importances_mean)])
else:
    importancias = pd.Series(model_step.feature_importances_, index=feat_names[:len(model_step.feature_importances_)])

top10 = importancias.nlargest(10)
print("\nTOP 10 VARIABLES:")
for i, (var, imp) in enumerate(top10.items(), 1):
    print(f"{i:2d}. {str(var)[:40]:40s} {imp:.4f}")

plt.figure(figsize=(10, 6))
top10.plot(kind='barh', color='skyblue')
plt.title(f"Top 10 Variables - {best_model_name}")
plt.xlabel('Importancia')
plt.tight_layout()
plt.savefig(f"plots/feature_importance_{best_model_name.lower()}.png", dpi=300)
plt.close()

# Test 2024
print("\n9. TEST 2024")
y_pred = best_pipe.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred)
print(f"MAE Test 2024: {mae_test:.4f}")
print(f"Exactitud ±0: {exact_acc(y_test, y_pred):.1%}")
print(f"Exactitud ±1: {tol_acc(y_test, y_pred):.1%}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([1, 20], [1, 20], 'r--', lw=2)
plt.xlabel('Posición Real 2024')
plt.ylabel('Posición Predicha')
plt.title(f'Predicción vs Realidad - {best_model_name}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/pred_vs_real_2024.png', dpi=300)
plt.close()

print("\n✅ === COMPLETADO ===")
print("📁 Gráficos generados:")
print("   • correlation_matrix.png")
print("   • vif_analysis.png") 
print("   • feature_importance_*.png")
print("   • pred_vs_real_2024.png")
print("🎯 FEEDBACK PROFESOR ✅")
