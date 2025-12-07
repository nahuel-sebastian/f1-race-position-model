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