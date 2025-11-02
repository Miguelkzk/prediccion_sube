import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Proyecto Predicción SUBE",
    page_icon="🚌",
    layout="wide"
)


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "modelo_pipeline_complejo.pkl")
PROMEDIOS_PATH = os.path.join(BASE_DIR, "datos_promedio.pkl")
CSV_PATH = os.path.join(BASE_DIR, "sube_clima_final_Mendoza.csv")

# --- 2. CONSTANTES Y MAPEOS ---
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

# --- 3. FUNCIONES DE CARGA ---

@st.cache_resource
def load_models():
    """Carga los modelos (pipeline y promedios) una sola vez."""
    try:
        # Usa las rutas absolutas
        pipeline = joblib.load(MODEL_PATH)
        promedios_data = joblib.load(PROMEDIOS_PATH)
        return pipeline, promedios_data
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_csv_data(csv_path):
    """Carga el CSV crudo una sola vez."""
    try:
        df = pd.read_csv(csv_path)
        df['Dia'] = pd.to_datetime(df['Dia'])
        if 'Condicion_Adversa' not in df.columns and 'Condicion_Cielo' in df.columns:
            df['Condicion_Adversa'] = df['Condicion_Cielo'].isin([7, 8, 9, 5, 6, 10, 11, 25, 26, 27]).astype(int)
        elif 'Condicion_Adversa' not in df.columns:
            df['Condicion_Adversa'] = 0 # Fallback
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

pipeline, promedios_data = load_models()
df_viz_raw = load_csv_data(CSV_PATH)


# Variables globales para predicción
if promedios_data:
    promedios_df = promedios_data['promedios']
    media_global = promedios_data['media_global']
    promedios_lookup = promedios_df.set_index(['Linea', 'Dia_Semana'])
    LINEAS_EJEMPLO = sorted(promedios_df['Linea'].unique())
else:
    promedios_df, media_global, promedios_lookup = None, 0, None
    LINEAS_EJEMPLO = ["Error: Cargar Modelos"]

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.prediccion_final = 0
    st.session_state.inputs = {}


st.title("🚌 Proyecto de Predicción de Pasajeros")

tab_info, tab_pred, tab_viz = st.tabs([
    "Sobre el Proyecto",
    "Predicción",
    "Visualizaciones"
])


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

with tab_pred:
    st.header("Formulario de Predicción")

    if pipeline is None or promedios_data is None:
        st.error("Error: Archivos de modelo ('*.pkl') no encontrados.")
        st.info("Por favor, ejecuta primero tu notebook para generar los artefactos.")

    elif not st.session_state.prediction_made:
        st.markdown("Completá los parámetros para estimar la cantidad de pasajeros.")
        with st.form(key="prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                linea = st.selectbox('Línea', options=LINEAS_EJEMPLO, key="linea")
                dia_semana_es = st.selectbox('Día de la Semana', options=list(DIAS_MAP.keys()), key="dia_semana")
                mes_es = st.selectbox('Mes', options=list(MESES_MAP.keys()), key="mes")
                feriado = st.selectbox('¿Es Feriado?', options=[0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No', key="feriado")
            with col2:
                condicion_adversa_str = st.selectbox('Precipitación', options=CONDICIONES_ADVERSAS_EJEMPLO, key="condicion_adversa")
                humedad_media = st.slider('Humedad Media (%)', 0.0, 100.0, 50.0, 1.0, key="humedad_media")
                temp_media = st.slider('Temperatura Media (°C)', -10.0, 40.0, 20.0, 0.5, key="temp_media")
                temp_min = st.slider('Temperatura Mínima (°C)', -15.0, 35.0, 15.0, 0.5, key="temp_min")
                temp_max = st.slider('Temperatura Máxima (°C)', -5.0, 45.0, 25.0, 0.5, key="temp_max")
                vel_viento = st.slider('Velocidad Media del Viento (km/h)', 0.0, 70.0, 15.0, 1.0, key="vel_viento")

            _, col_btn_center, _ = st.columns([2, 1, 2]) # Columnas para centrar el botón
            with col_btn_center:
                submit_button = st.form_submit_button(label='Predecir Cantidad', use_container_width=True)

        if submit_button:
            mes_num = MESES_MAP[mes_es]
            dia_eng = DIAS_MAP[dia_semana_es]
            st.session_state.inputs = {
                "Línea": linea, "Día de la Semana": dia_semana_es, "Mes": mes_es,
                "Feriado": "Sí" if feriado == 1 else "No", "Precipitación": condicion_adversa_str,
                "Humedad Media": humedad_media, "Temp. Media": temp_media,
                "Temp. Mínima": temp_min, "Temp. Máxima": temp_max, "Velocidad Viento": vel_viento
            }
            try: avg_cantidad = promedios_lookup.loc[(linea, dia_eng)]['Cantidad']
            except KeyError: avg_cantidad = media_global
            input_data = {feature: [np.nan] for feature in ALL_FEATURES}
            condicion_adversa_bool = 0 if condicion_adversa_str == 'Ninguna' else 1
            input_data.update({
                'Linea': [linea], 'Dia_Semana': [dia_eng], 'Mes': [mes_num], 'Feriado': [feriado],
                'Condicion_Adversa': [condicion_adversa_bool], 'Humedad_Media': [humedad_media],
                'Temp_media': [temp_media], 'Temp_min': [temp_min], 'Temp_max': [temp_max],
                'Vel_Prom_Viento': [vel_viento], 'Cantidad_lag_1': [avg_cantidad],
                'Cantidad_lag_7': [avg_cantidad], 'Cantidad_ma_7': [avg_cantidad]
            })
            input_df = pd.DataFrame(input_data)
            for col in CAT_FEATURES: input_df[col] = input_df[col].astype(str)
            prediccion_log = pipeline.predict(input_df)
            prediccion_final = np.expm1(prediccion_log)
            st.session_state.prediccion_final = prediccion_final[0]
            st.session_state.prediction_made = True
            st.rerun()

    elif st.session_state.prediction_made:
        st.header("Resultado de la Predicción")
        st.metric(label="Pasajeros Predichos (Cantidad) para la ciudad de Mendoza.", value=f"{int(st.session_state.prediccion_final):,}")
        st.divider()
        st.subheader("Parámetros Utilizados")
        col1, col2 = st.columns(2)
        inputs = st.session_state.inputs
        with col1:
            st.metric(label="Línea", value=inputs["Línea"])
            st.metric(label="Día de la Semana", value=inputs["Día de la Semana"])
            st.metric(label="Mes", value=inputs["Mes"])
            st.metric(label="Feriado", value=inputs["Feriado"])
            st.metric(label="Precipitación", value=inputs["Precipitación"])
        with col2:
            st.metric(label="Humedad", value=f"{inputs['Humedad Media']}%")
            st.metric(label="Temp. Media", value=f"{inputs['Temp. Media']}°C")
            st.metric(label="Temp. Mínima", value=f"{inputs['Temp. Mínima']}°C")
            st.metric(label="Temp. Máxima", value=f"{inputs['Temp. Máxima']}°C")
            st.metric(label="Vel. Media del Viento", value=f"{inputs['Velocidad Viento']} km/h")
        _, col_btn_center, _ = st.columns([1, 1, 1])
        with col_btn_center:
            if st.button("Hacer otra predicción", use_container_width=True):
                st.session_state.prediction_made = False
                st.rerun()

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
            chart_dia = alt.Chart(df_dia).mark_bar().encode(
                x=alt.X('Dia_Semana', sort=dias_orden, title='Día de la Semana'),
                y=alt.Y('Cantidad', title='Pasajeros Promedio'),
                tooltip=['Dia_Semana', 'Cantidad']
            ).interactive()
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
        with col_clima2:
            st.markdown("**Temperatura Media vs. Pasajeros**")
            st.markdown("Este gráfico muestra la tendencia de viajes según la temperatura. **La gente viaja más en días templados.**")

            chart_temp = alt.Chart(df_temp_agg).mark_line(point=True).encode(
                x=alt.X('Temp_Redondeada', title='Temperatura Media (°C)', scale=alt.Scale(zero=False)),
                y=alt.Y('Cantidad', title='Pasajeros Promedio'),
                tooltip=['Temp_Redondeada', 'Cantidad']
            ).interactive()
            st.altair_chart(chart_temp, use_container_width=True)