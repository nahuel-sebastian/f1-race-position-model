# MODELO EXPLICATIVO DE PREDICCIÓN DE POSICIÓN EN CARRERA DE F1 (2018-2024)

## Descripción
Este proyecto analiza datos de paradas en boxes de Fórmula 1 entre 2018 y 2024. Utiliza técnicas de preprocesamiento, modelos de machine learning y análisis de interpretabilidad para predecir posiciones finales y entender los factores más influyentes en el rendimiento de los pilotos.

## Tecnologías utilizadas
- **Python 3.10+**: Lenguaje principal para el análisis y modelado.
- **Bibliotecas principales**:
  - `pandas`, `numpy`: Manipulación y análisis de datos.
  - `matplotlib`, `seaborn`: Visualización de datos.
  - `scikit-learn`: Preprocesamiento, modelado y evaluación.
  - `xgboost`, `shap`: Modelos avanzados y análisis de interpretabilidad.
  - `statsmodels`: Cálculo de VIF para análisis de multicolinealidad.

## Estructura del proyecto

```plaintext
PRACTICA_FINAL_IAO/
├── code/                               # Script principal del proyecto
│   └── code.py                         # Código ejecutable
├── data/                               # Datos de entrada
│   └── f1_pitstops_2018_2024.csv       # Dataset principal
├── plots/                              # Gráficos y resultados generados
├── docs/                               # Documentación adicional
└── README.md                           # Descripción general del proyecto
```

## Requisitos previos
1. **Python**: Asegúrate de tener Python 3.10 o superior instalado.

2. **Creación del entorno virtual**:
   ```bash
   # Entorno virtual
   python -m venv venv
   # Windows
   venv/Scripts/activate
   # Linux
   source venv/bin/activate
   # Desactivar entorno virtual
   deactivate
   ```
3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución
### 1. Preprocesamiento y modelado
Ejecuta el script principal para entrenar modelos y evaluar resultados:
```bash
python3 code/code.py
```
Los gráficos se guardan en la carpeta `plots/`, mientras que los resultados se muestran en la terminal como prints.