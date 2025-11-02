# Predicción de Pasajeros SUBE

## Descripción del Proyecto

Este proyecto utiliza un **modelo de Machine Learning (LightGBM)** para predecir la cantidad de pasajeros del transporte público en Mendoza.

La aplicación web, construida con Streamlit, permite:

1.  **Predecir** la cantidad de pasajeros futuros basándose en 10 parámetros de entrada (Línea, Clima, Fecha).
2.  **Visualizar** los datos históricos para entender las tendencias y la relación entre el clima y la demanda.

## Estructura del Proyecto

```
dev/
├── app.py                     # Aplicación Streamlit unificada (con pestañas)
├── Copy of ... .ipynb         # Notebook de entrenamiento (genera los .pkl)
├── modelo_pipeline_complejo.pkl # (Generado) Pipeline de Preprocesamiento + Modelo LGBM
├── datos_promedio.pkl         # (Generado) Datos para estimación de lags
├── sube_clima_final_Mendoza (2).csv # Datos crudos para visualizaciones
├── requirements.txt           # Dependencias
```

