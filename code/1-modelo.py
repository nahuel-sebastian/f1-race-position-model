import pandas as pd
import numpy as np

seed = 6967

datos = pd.read_csv("data/f1_pitstops_2018_2024.csv", sep=",")

datos.info()
print(f"Tamaño del dataset: {datos.shape}")
print(f"Número total de elementos del dataset: {datos.size}")

# Location, country, date, round, race name, time_of_race, Total Pit Stops

datos.drop(columns=[
    "Location", "Country", "Date", "Round", "Race Name", "Time_of_race", "Total Pit Stops"
], inplace=True)

nulos = datos.isnull().sum()
print(nulos[nulos > 0])

mask_final_stint = datos['Pit_Time'] == "Final Stint"
datos.loc[mask_final_stint, 'Pit_Lap'] = datos.loc[mask_final_stint, 'Pit_Lap'].fillna(-1)
print(datos['Pit_Lap'])

nulos = datos.isnull().sum()
print(nulos[nulos > 0])

datos.info()