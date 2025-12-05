import pandas as pd
import numpy as np

seed = 6967

datos = pd.read_csv("data/f1_pitstops_2018_2024.csv", sep=",")

print("Información inicial del dataset:")
datos.info()
print(f"Tamaño del dataset: {datos.shape}")
print(f"Número total de elementos del dataset: {datos.size}")

# Location, country, date, round, race name, time_of_race, Total Pit Stops
print("Eliminación de columnas de contexto poco útiles para la predicción directa:")
datos.drop(columns=[
    "Location", "Country", "Date", "Round", "Race Name", "Time_of_race", "Total Pit Stops"
], inplace=True)

print("\nNulos antes de tratar Pit_Lap:")
nulos = datos.isnull().sum()
print(nulos[nulos > 0])

# Aseguramos que los vacíos de Pit_Lap sean NaN
datos['Pit_Lap'] = datos['Pit_Lap'].replace('', np.nan)

# Rellenar Pit_Lap con -1 solo en el stint final
mask_final_stint = datos['Pit_Time'] == "Final Stint"
datos.loc[mask_final_stint, 'Pit_Lap'] = datos.loc[mask_final_stint, 'Pit_Lap'].fillna(-1)

# Convertir a numérico entero (si quieres trabajar siempre con enteros)
datos['Pit_Lap'] = pd.to_numeric(datos['Pit_Lap'], errors='coerce')

print("\nNulos después de tratar Pit_Lap:")
nulos = datos.isnull().sum()
print(nulos[nulos > 0])

print("\nInformación del dataset tras preprocesado inicial:")
datos.info()