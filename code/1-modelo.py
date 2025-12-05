import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

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
        "Race Name", "Time_of_race", "Total Pit Stops"
    ],
    inplace=True
)

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


# ============================
# 6. EDA básico orientado a Position
# ============================

# Histograma de la variable objetivo: Position
plt.figure(figsize=(6, 4))
plt.hist(datos['Position'], bins=20)
plt.title('Distribución de Position')
plt.xlabel('Position')
plt.ylabel('Frecuencia')
plt.show()

# Histogramas de algunas variables numéricas clave
for v in ['TotalPitStops', 'AvgPitStopTime', 'Stint Length']:
    if v in datos.columns:
        plt.figure(figsize=(6,4))
        plt.hist(datos[v], bins=20)
        plt.title(f'Distribución de {v}')
        plt.xlabel(v)
        plt.ylabel('Frecuencia')
        plt.show()

# Scatter de variables clave frente a Position
for v in ['TotalPitStops', 'AvgPitStopTime', 'Stint Length']:
    if v in datos.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(datos[v], datos['Position'], alpha=0.3)
        plt.title(f'{v} vs Position')
        plt.xlabel(v)
        plt.ylabel('Position')
        plt.gca().invert_yaxis()  # 1 es mejor que 20
        plt.show()

# Correlación de cada numérica con Position
corr_pos = datos[num_cols].corr()['Position'].sort_values(ascending=False)
print("\nCorrelación de cada variable numérica con Position:")
print(corr_pos)

# =======================
# 7. Matriz de correlación
# =======================

num_cols = datos.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = datos[num_cols].corr().values

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(num_cols)))
ax.set_yticks(np.arange(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=90)
ax.set_yticklabels(num_cols)

fig.colorbar(im, ax=ax)
plt.title('Matriz de correlación entre variables numéricas')
plt.tight_layout()
plt.show()

# Pares de variables con alta correlación absoluta (> 0.8)
corr_df = datos[num_cols].corr().abs()
high_corr = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
print("\nPares de variables con correlación > 0.8:")
print(high_corr[high_corr > 0.8].stack().sort_values(ascending=False))

# 1. Separar las filas por año (Temporal Split)
train_df = datos[datos['Season'].between(2018, 2023)].copy()
test_df  = datos[datos['Season'] == 2024].copy()

print(f"Filas para entrenar (2018-2023): {train_df.shape[0]}")
print(f"Filas para validar (2024): {test_df.shape[0]}")

# --- NUEVA SECCIÓN: PREPARACIÓN DE DATOS (NUMÉRICOS + TEXTO) ---

# 1. Definir nuestras variables (Features)
# Variables numéricas que queremos usar (¡incluyendo el clima!)
columnas_num = ['TotalPitStops', 'AvgPitStopTime', 'Air_Temp_C',
                'Track_Temp_C', 'Humidity_%', 'Wind_Speed_KMH', 'Pit_Lap']

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