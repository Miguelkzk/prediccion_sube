# --- app.py (Versión 4.7 - Filtro de Línea + Correcciones) ---

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import os
from sklearn.metrics import mean_absolute_error

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Proyecto Predicción SUBE",
    page_icon="🚌",
    layout="wide"
)

# --- Rutas Absolutas (Sin cambios) ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "modelo_pipeline_complejo.pkl")
PROMEDIOS_PATH = os.path.join(BASE_DIR, "datos_promedio.pkl")
CSV_PATH = os.path.join(BASE_DIR, "sube_clima_final_Mendoza.csv")

# --- 2. CONSTANTES Y MAPEOS (Sin cambios) ---
ALL_FEATURES = [
    'Linea', 'Nombre_Empresa', 'Provincia', 'Municipio',
    'Temp_media', 'Temp_max', 'Temp_min', 'Lluvia_Binaria', 'Precip_Total',
    'Humedad_Media', 'Vel_Prom_Viento', 'Condicion_Adversa', 'Mes', 'Dia_Semana', 'Feriado',
    'Es_FinDeSemana', 'Temp_Templada', 'Temp_Extrema', 'Adversa_Finde', 'Feriado_TempExtrema',
    'Cantidad_lag_1', 'Cantidad_lag_7', 'Cantidad_ma_7'
]
CAT_FEATURES = ['Linea', 'Nombre_Empresa', 'Provincia', 'Municipio', 'Dia_Semana']
MESES_MAP = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
    "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}
DIAS_MAP = {
    "Lunes": "Monday", "Martes": "Tuesday", "Miércoles": "Wednesday",
    "Jueves": "Thursday", "Viernes": "Friday", "Sábado": "Saturday", "Domingo": "Sunday"
}
CONDICIONES_ADVERSAS_EJEMPLO = [
    'Ninguna', 'Lluvia', 'Lluvia ligera', 'Lluvia intensa', 'Niebla', 'Niebla helada',
    'Lluvia helada', 'Lluvia helada intensa', 'Tormenta eléctrica', 'Tormenta eléctrica intensa', 'Tormenta'
]

# --- 3. FUNCIONES DE CARGA (Corregidas) ---

@st.cache_resource
def load_models():
    """Carga los modelos (pipeline y promedios) una sola vez."""
    try:
        pipeline = joblib.load(MODEL_PATH)
        promedios_data = joblib.load(PROMEDIOS_PATH)
        return pipeline, promedios_data
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_csv_data(csv_path):
    """Carga el CSV crudo y aplica la INGENIERÍA DE FEATURES una sola vez."""
    try:
        df = pd.read_csv(csv_path)
        df['Dia'] = pd.to_datetime(df['Dia'])

        # --- INICIO DE INGENIERÍA DE FEATURES (de Celda 4) ---
        if 'Condicion_Cielo' in df.columns:
            df['Condicion_Adversa'] = df['Condicion_Cielo'].isin([7, 8, 9, 5, 6, 10, 11, 25, 26, 27]).astype(int)
        else:
            df['Condicion_Adversa'] = 0

        df['Mes'] = df['Dia'].dt.month
        df['Es_FinDeSemana'] = df['Dia_Semana'].isin(['Saturday', 'Sunday']).astype(int)
        df['Temp_Templada'] = df['Temp_media'].between(8,28).astype(int)
        df['Temp_Extrema'] = ((df['Temp_media'] < 8) | (df['Temp_media'] > 28)).astype(int)
        df['Adversa_Finde'] = df['Condicion_Adversa'] * df['Es_FinDeSemana']
        df['Feriado_TempExtrema'] = df['Feriado'] * df['Temp_Extrema']

        df = df.sort_values(['Linea','Dia'])
        df['Cantidad_lag_1'] = df.groupby('Linea')['Cantidad'].shift(1)
        df['Cantidad_lag_7'] = df.groupby('Linea')['Cantidad'].shift(7)
        df['Cantidad_ma_7'] = df.groupby('Linea')['Cantidad'].transform(lambda x: x.rolling(7, min_periods=1).mean())

        for col in ['Cantidad_lag_1', 'Cantidad_lag_7', 'Cantidad_ma_7']:
            df[col] = df.groupby('Linea')[col].transform(lambda x: x.fillna(x.mean()))

        # Rellenar NaNs numéricos y categóricos restantes
        try:
            df = df.fillna(df.mean(numeric_only=True))
        except TypeError:
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df[cat_cols] = df[cat_cols].fillna('missing')
        # --- FIN DE INGENIERÍA DE FEATURES ---

        return df
    except FileNotFoundError:
        return None

@st.cache_data
def get_viz_dataframes(df_raw):
    """
    Realiza TODOS los cálculos pesados de Pandas una sola vez
    y devuelve DataFrames pequeños listos para graficar.
    """
    df = df_raw.copy()

    # 1. Datos para Gráfico Semanal
    df_semanal = df.set_index('Dia').resample('W')['Cantidad'].sum().reset_index()
    # 2. Datos para Gráfico por Línea
    df_linea = df.groupby('Linea')['Cantidad'].mean().reset_index()
    # 3. Datos para Gráfico por Día
    df_dia = df.groupby('Dia_Semana')['Cantidad'].mean().reset_index()
    # 4. Datos para Histograma (Optimizado)
    q_99 = df['Cantidad'].quantile(0.99)
    counts, bins = np.histogram(df['Cantidad'], bins=100, range=(0, q_99))
    hist_data = pd.DataFrame({
        'Frecuencia': counts, 'Rango_Inicio': bins[:-1], 'Rango_Fin': bins[1:]
    })
    hist_data['Rango_Etiqueta'] = hist_data.apply(lambda r: f"{int(r['Rango_Inicio'])}-{int(r['Rango_Fin'])}", axis=1)
    # 5. Datos para Gráfico de Días Adversos
    df['Tipo_Dia'] = df['Condicion_Adversa'].map({0: 'Día Normal', 1: 'Día Adverso'})
    df_adverso = df.groupby('Tipo_Dia')['Cantidad'].mean().reset_index()
    # 6. Datos para Gráfico de Temperatura
    df['Temp_Redondeada'] = df['Temp_media'].round()
    df_temp_agg = df.groupby('Temp_Redondeada')['Cantidad'].mean().reset_index()

    return df_semanal, df_linea, df_dia, hist_data, df_adverso, df_temp_agg

# --- 4. CARGAR DATOS Y MODELOS (Sin cambios) ---
pipeline, promedios_data = load_models()
df_viz_raw = load_csv_data(CSV_PATH)

# Variables globales para predicción
if promedios_data and df_viz_raw is not None:
    promedios_df = promedios_data['promedios']
    media_global = promedios_data['media_global']
    promedios_lookup = promedios_df.set_index(['Linea', 'Dia_Semana'])
    # Usamos las líneas del CSV cargado, que es más robusto
    LINEAS_EJEMPLO = sorted(df_viz_raw['Linea'].unique())
else:
    promedios_df, media_global, promedios_lookup = None, 0, None
    LINEAS_EJEMPLO = ["Error: Cargar Modelos"]

# --- 5. TÍTULO PRINCIPAL Y PESTAÑAS (Sin cambios) ---
st.title("🚌 Proyecto de Predicción de Pasajeros")

tab_info, tab_eval, tab_viz = st.tabs([
    "Sobre el Proyecto",
    "Evaluación del Modelo",
    "Visualizaciones"
])

# --- PESTAÑA 1: SOBRE EL PROYECTO (Sin cambios) ---
with tab_info:
    st.header("Sobre el Proyecto")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Integrantes del grupo")
        members = [
            "Juan Manuel Valdivia", "Lucio Malgioglio",
            "Lucianda Maldonado", "Miguel Kruzliak",
        ]
        for m in members: st.markdown(f"- {m}")
        st.image("https://www.mendoza.gov.ar/wp-content/uploads/2023/02/mendotran-sube-portada.jpg")
        st.subheader("Herramientas Utilizadas")
        st.markdown("""
        - **Python:** Lenguaje principal.
        - **Pandas:** Para manipulación y limpieza de datos.
        - **Scikit-Learn:** Para crear el pipeline de preprocesamiento.
        - **LightGBM:** Para el modelo de regresión.
        - **Streamlit:** Para construir esta aplicación web.
        - **Altair:** Para la creación de los gráficos interactivos.
        """)
    with col2:
        st.subheader("Objetivo")
        st.markdown("""
        El objetivo principal de este proyecto es aplicar técnicas de Machine Learning para
        crear un modelo capaz de predecir la demanda de cantidad del transporte
        público en Mendoza.
        """)
        st.subheader("Conjunto de Datos")
        st.markdown("""
        Se utilizó un conjunto de datos que combina dos fuentes:
        1.  **Datos de SUBE:** Registros diarios de transacciones por línea de colectivo.
        2.  **Datos Climáticos:** Información meteorológica (temperatura, humedad, viento, etc.)
            para los mismos días.
        """)
        st.subheader("Modelo de Machine Learning")
        st.markdown("""
        -   **Modelo:** `LightGBM Regressor`.
        -   **Features:** Se utilizaron 23 features.
        -   **Transformación:** Se aplicó una transformación `log1p` a la variable objetivo (`Cantidad`).
        -   **Pipeline:** Se construyó un `Pipeline` de `scikit-learn` que incluye un `ColumnTransformer`
            y el modelo `LGBM`.
        """)

# --- PESTAÑA 2: EVALUACIÓN DEL MODELO (¡Sección Modificada!) ---
with tab_eval:
    st.header("📈 Evaluación: Predicción vs. Datos Reales")

    if pipeline is None or df_viz_raw is None:
        st.error("Error: No se pudieron cargar los modelos o los datos CSV.")
        st.info("Asegúrate de que los archivos .pkl y .csv están en el repositorio.")
    else:
        st.markdown("""
        Esta pestaña te permite comparar el rendimiento del modelo contra los datos reales del dataset.
        Selecciona una línea (opcional) y cuántos de los últimos días quieres visualizar:
        """)

        col1, col2 = st.columns(2)

        with col1:
            linea_options = ["Todas las líneas (Agregado)"] + LINEAS_EJEMPLO
            linea_seleccionada = st.selectbox("Selecciona una Línea (opcional):", options=linea_options)

        with col2:
            # Usamos nunique() sobre los días del DF *cargado*
            max_dias = df_viz_raw['Dia'].nunique()
            if max_dias < 7:
                st.warning("El dataset es muy pequeño para esta evaluación.")
                n_dias = 0
            else:
                n_dias = st.number_input(
                    "Cantidad de últimos días a comparar:",
                    min_value=7,
                    max_value=max_dias,
                    value=min(30, max_dias),
                    step=1
                )

        if n_dias > 0:
            with st.spinner(f"Calculando predicciones para '{linea_seleccionada}'..."):
                try:
                    # 1. Data Subsetting (Correcto)
                    last_n_dates = df_viz_raw['Dia'].sort_values(ascending=False).unique()[:n_dias][::-1]
                    df_period = df_viz_raw[df_viz_raw['Dia'].isin(last_n_dates)].copy()

                    # 2. Filtrado Opcional
                    if linea_seleccionada == "Todas las líneas (Agregado)":
                        st.info("Mostrando la suma agregada de todas las líneas para los últimos N días.")
                        X_eval = df_period.copy()
                    else: # Caso de una línea específica
                        st.info(f"Mostrando la evaluación para: {linea_seleccionada}")
                        X_eval = df_period[df_period['Linea'] == linea_seleccionada].copy()

                        if X_eval.empty:
                            st.warning(f"No se encontraron datos para '{linea_seleccionada}' en los últimos {n_dias} días.")
                            st.stop()

                    # 3. Pre-procesamiento (¡Importante!)
                    # Rellenar NaNs categóricos y numéricos ANTES de predecir
                    # El pipeline de sklearn es sensible a NaNs incluso si tiene un imputer
                    for col in CAT_FEATURES:
                        X_eval[col] = X_eval[col].fillna('missing').astype(str)

                    numeric_cols = X_eval.select_dtypes(include=np.number).columns
                    # Llenamos con 0 para que la suma/agregación no se vea afectada por NaNs
                    X_eval[numeric_cols] = X_eval[numeric_cols].fillna(0)

                    # 4. Predicción
                    prediccion_log = pipeline.predict(X_eval)
                    X_eval['Prediccion'] = np.expm1(prediccion_log)
                    X_eval['Valor Real'] = X_eval['Cantidad']

                    # 5. Agregación (¡Crítico para gráficos limpios!)
                    # Suma por día. Esto funciona para "Todas" y para "Una línea"
                    df_agg = X_eval.groupby('Dia')[['Valor Real', 'Prediccion']].sum().reset_index()
                    df_grafico = df_agg.rename(columns={'Prediccion': 'Predicción del Modelo'})

                    # 6. Preparar datos para el gráfico
                    df_grafico['Fecha Real'] = df_grafico['Dia'].dt.strftime('%Y-%m-%d')
                    df_grafico['Dia (Nro)'] = np.arange(1, len(df_grafico) + 1)

                    # --- ¡CORRECCIÓN DEL MELT! ---
                    df_melted = df_grafico.melt(
                        id_vars=['Dia (Nro)', 'Fecha Real'],
                        value_vars=['Valor Real', 'Predicción del Modelo'], # Ser explícito
                        var_name='Tipo de Valor',
                        value_name='Cantidad_Melted' # Usar un nombre de valor nuevo
                    )

                    # 7. Gráfico de Altair
                    st.subheader(f"Comparación de los Últimos {n_dias} Días ({linea_seleccionada})")

                    chart = alt.Chart(df_melted).mark_line(point=True).encode(
                        x=alt.X('Dia (Nro)', title=f'Últimos {n_dias} Días (en orden)', axis=alt.Axis(format='d')),
                        # Usar el nuevo value_name 'Cantidad_Melted'
                        y=alt.Y('Cantidad_Melted', title='Cantidad de Pasajeros'),
                        color=alt.Color('Tipo de Valor', title="Valor:"),
                        tooltip=[
                            alt.Tooltip('Dia (Nro)', title="Día Nro."),
                            alt.Tooltip('Fecha Real'),
                            'Tipo de Valor',
                            # Usar el nuevo value_name 'Cantidad_Melted'
                            alt.Tooltip('Cantidad_Melted', title="Cantidad", format=',.0f')
                        ]
                    ).interactive()

                    st.altair_chart(chart, use_container_width=True)

                    # 8. Métrica de Error (MAE)
                    mae = mean_absolute_error(df_grafico['Valor Real'], df_grafico['Predicción del Modelo'])
                    st.metric(
                        label=f"Error Absoluto Medio (MAE) para este período",
                        value=f"{mae:,.2f} pasajeros"
                    )
                    st.info(f"El MAE indica que, en promedio, las predicciones del modelo para este período se desvían en {mae:,.2f} pasajeros del valor real.")

                except Exception as e:
                    st.error(f"Ocurrió un error al generar la predicción: {e}")
                    st.exception(e)

# --- PESTAÑA 3: VISUALIZACIONES (Sin cambios) ---
with tab_viz:
    st.header("Hallazgos y Visualizaciones")

    if df_viz_raw is None:
        st.error(f"Error: No se encontró el archivo en la ruta: {CSV_PATH}")
        st.info("Asegúrate de que el archivo CSV esté en la carpeta 'dev/' y subido a GitHub.")
    else:
        st.markdown(f"Exploración de los datos históricos (`{os.path.basename(CSV_PATH)}`)")

        with st.spinner("Procesando gráficos por primera vez..."):
            df_semanal, df_linea, df_dia, hist_data, df_adverso, df_temp_agg = get_viz_dataframes(df_viz_raw)

        st.subheader("1. Serie Temporal de Pasajeros (Agregado Semanal)")
        st.markdown("Agregamos la cantidad total de pasajeros por semana para ver la tendencia general, patrones estacionales y el impacto de eventos como la pandemia.")

        chart_semanal = alt.Chart(df_semanal).mark_line().encode(
            x=alt.X('Dia', title='Fecha'),
            y=alt.Y('Cantidad', title='Pasajeros Totales (por Semana)'),
            tooltip=['Dia', 'Cantidad']
        ).interactive()
        st.altair_chart(chart_semanal, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2. Pasajeros Promedio por Línea")
            chart_linea = alt.Chart(df_linea).mark_bar().encode(
                x=alt.X('Linea', sort='-y', title='Línea'),
                y=alt.Y('Cantidad', title='Pasajeros Promedio'),
                tooltip=['Linea', 'Cantidad']
            ).interactive()
            st.altair_chart(chart_linea, use_container_width=True)
        with col2:
            st.subheader("3. Pasajeros Promedio por Día")

            dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            base = alt.Chart(df_dia).encode(
                y=alt.Y('Dia_Semana', sort=dias_orden, title='Día de la Semana'),
                x=alt.X('Cantidad', title='Pasajeros Promedio'),
                tooltip=['Dia_Semana', alt.Tooltip('Cantidad', format=',.0f')]
            )

            bars = base.mark_bar()

            text = base.mark_text(
                align='left',
                baseline='middle',
                dx=3
            ).encode(
                text=alt.Text('Cantidad', format=',.0f'),
                color=alt.value('black')
            )

            chart_dia = (bars + text).interactive()

            st.altair_chart(chart_dia, use_container_width=True)

        st.subheader("4. Distribución de la variable 'Cantidad'")
        st.markdown("La distribución está fuertemente sesgada a la derecha. Esta es la razón por la cual en el notebook se aplica una transformación logarítmica (`log1p`) antes de entrenar.")

        hist_cantidad = alt.Chart(hist_data).mark_bar().encode(
            x=alt.X('Rango_Etiqueta', sort=None, title='Cantidad de Pasajeros (bins)'),
            y=alt.Y('Frecuencia', title='Frecuencia'),
            tooltip=['Rango_Etiqueta', 'Frecuencia']
        ).interactive()
        st.altair_chart(hist_cantidad, use_container_width=True)

        st.divider()
        st.subheader("5. Relación Clima-Pasajeros")
        col_clima1, col_clima2 = st.columns(2)
        with col_clima1:
            st.markdown("**Impacto de Precipitación Adversa**")
            st.markdown("Se compara el promedio de pasajeros en días con precipitación (Lluvia, Tormenta, Niebla) vs. días normales. **El mal tiempo reduce la cantidad de viajeros.**")

            chart_adverso = alt.Chart(df_adverso).mark_bar().encode(
                x=alt.X('Tipo_Dia', title='Condición del Día', sort='-y'),
                y=alt.Y('Cantidad', title='Pasajeros Promedio'),
                color='Tipo_Dia',
                tooltip=['Tipo_Dia', 'Cantidad']
            ).interactive()
            st.altair_chart(chart_adverso, use_container_width=True)
        with col2:
            st.markdown("**Temperatura Media vs. Pasajeros**")
            st.markdown("Este gráfico muestra la tendencia de viajes según la temperatura. **La gente viaja más en días templados.**")

            chart_temp = alt.Chart(df_temp_agg).mark_line(point=True).encode(
                x=alt.X('Temp_Redondeada', title='Temperatura Media (°C)', scale=alt.Scale(zero=False)),
                y=alt.Y('Cantidad', title='Pasajeros Promedio'),
                tooltip=['Temp_Redondeada', 'Cantidad']
            ).interactive()
            st.altair_chart(chart_temp, use_container_width=True)