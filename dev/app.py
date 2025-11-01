import streamlit as st
import pandas as pd
import numpy as np
import joblib

ALL_FEATURES = [
    'Linea', 'Nombre_Empresa', 'Provincia', 'Municipio',
    'Temp_media', 'Temp_max', 'Temp_min', 'Lluvia_Binaria', 'Precip_Total',
    'Humedad_Media', 'Vel_Prom_Viento', 'Condicion_Adversa', 'Mes', 'Dia_Semana', 'Feriado',
    'Es_FinDeSemana', 'Temp_Templada', 'Temp_Extrema', 'Adversa_Finde', 'Feriado_TempExtrema',
    'Cantidad_lag_1', 'Cantidad_lag_7', 'Cantidad_ma_7'
]
CAT_FEATURES = ['Linea', 'Nombre_Empresa', 'Provincia', 'Municipio', 'Dia_Semana']

try:
    pipeline = joblib.load("modelo_pipeline_complejo.pkl")
    promedios_data = joblib.load("datos_promedio.pkl")
    promedios_df = promedios_data['promedios']
    media_global = promedios_data['media_global']
    promedios_lookup = promedios_df.set_index(['Linea', 'Dia_Semana'])
except FileNotFoundError:
    st.error("Archivos de modelo ('*.pkl') no encontrados.")
    st.info("Por favor, ejecuta primero tu notebook para generar los artefactos.")
    st.stop()


# --- 2. Configuración de la Página y Constantes ---
st.set_page_config(page_title="Predicción SUBE", page_icon="🚌", layout="centered")
st.title('🚌 Predicción de Pasajeros')

# Mapeo para traducciones
MESES_MAP = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
    "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}
DIAS_MAP = {
    "Lunes": "Monday", "Martes": "Tuesday", "Miércoles": "Wednesday",
    "Jueves": "Thursday", "Viernes": "Friday", "Sábado": "Saturday", "Domingo": "Sunday"
}

# Listas para los selectores
LINEAS_EJEMPLO = sorted(promedios_df['Linea'].unique())
CONDICIONES_ADVERSAS_EJEMPLO = [
    'Ninguna', 'Lluvia', 'Lluvia ligera', 'Lluvia intensa', 'Niebla', 'Niebla helada',
    'Lluvia helada', 'Lluvia helada intensa', 'Tormenta eléctrica', 'Tormenta eléctrica intensa', 'Tormenta'
]

# Inicializar el "estado" de la aplicación
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.prediccion_final = 0
    st.session_state.inputs = {}



# --- ESTADO 1: MOSTRAR EL FORMULARIO DE ENTRADA ---
if not st.session_state.prediction_made:

    st.markdown("Completá los parámetros para estimar la cantidad de pasajeros.")

    with st.form(key="prediction_form"):

        col1, col2 = st.columns(2)

        # --- Columna 1 ---
        with col1:
            linea = st.selectbox('Línea', options=LINEAS_EJEMPLO, key="linea")
            dia_semana_es = st.selectbox('Día de la Semana', options=list(DIAS_MAP.keys()), key="dia_semana")
            mes_es = st.selectbox('Mes', options=list(MESES_MAP.keys()), key="mes")
            feriado = st.selectbox('¿Es Feriado?', options=[0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No', key="feriado")

        # --- Columna 2 ---
        with col2:
            condicion_adversa_str = st.selectbox('Precipitación', options=CONDICIONES_ADVERSAS_EJEMPLO, key="condicion_adversa")
            humedad_media = st.slider('Humedad Media (%)', 0.0, 100.0, 50.0, 1.0, key="humedad_media")
            temp_media = st.slider('Temperatura Media (°C)', -10.0, 40.0, 20.0, 0.5, key="temp_media")
            temp_min = st.slider('Temperatura Mínima (°C)', -15.0, 35.0, 15.0, 0.5, key="temp_min")
            temp_max = st.slider('Temperatura Máxima (°C)', -5.0, 45.0, 25.0, 0.5, key="temp_max")
            vel_viento = st.slider('Velocidad Media del Viento (km/h)', 0.0, 70.0, 15.0, 1.0, key="vel_viento")

        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submit_button = st.form_submit_button(label='Predecir Cantidad', use_container_width=True)

    if submit_button:
        mes_num = MESES_MAP[mes_es]
        dia_eng = DIAS_MAP[dia_semana_es]

        st.session_state.inputs = {
            "Línea": linea,
            "Día de la Semana": dia_semana_es,
            "Mes": mes_es,
            "Feriado": "Sí" if feriado == 1 else "No",
            "Precipitación": condicion_adversa_str,
            "Humedad Media": humedad_media,
            "Temp. Media": temp_media,
            "Temp. Mínima": temp_min,
            "Temp. Máxima": temp_max,
            "Velocidad Viento": vel_viento
        }

        # --- 4.1. Estimación de Lags ---
        try:
            avg_cantidad = promedios_lookup.loc[(linea, dia_eng)]['Cantidad']
        except KeyError:
            avg_cantidad = media_global

        # --- 4.2. Construir el DataFrame de Inferencia ---
        input_data = {feature: [np.nan] for feature in ALL_FEATURES}

        condicion_adversa_bool = 0 if condicion_adversa_str == 'Ninguna' else 1

        # Llenar con los datos del formulario
        input_data['Linea'] = [linea]
        input_data['Dia_Semana'] = [dia_eng]
        input_data['Mes'] = [mes_num]
        input_data['Feriado'] = [feriado]
        input_data['Condicion_Adversa'] = [condicion_adversa_bool] # El modelo aún usa el nombre original
        input_data['Humedad_Media'] = [humedad_media]
        input_data['Temp_media'] = [temp_media]
        input_data['Temp_min'] = [temp_min]
        input_data['Temp_max'] = [temp_max]
        input_data['Vel_Prom_Viento'] = [vel_viento]

        # Llenar Lags estimados
        input_data['Cantidad_lag_1'] = [avg_cantidad]
        input_data['Cantidad_lag_7'] = [avg_cantidad]
        input_data['Cantidad_ma_7'] = [avg_cantidad]

        input_df = pd.DataFrame(input_data)

        # --- 4.3. Corrección de Tipos ---
        for col in CAT_FEATURES:
            input_df[col] = input_df[col].astype(str)

        # --- 4.4. Predecir ---
        prediccion_log = pipeline.predict(input_df)
        prediccion_final = np.expm1(prediccion_log)

        # --- 4.5. Guardar estado y refrescar ---
        st.session_state.prediccion_final = prediccion_final[0]
        st.session_state.prediction_made = True
        st.rerun()


# --- ESTADO 2: MOSTRAR LOS RESULTADOS ---
if st.session_state.prediction_made:

    st.header("Resultado de la Predicción")
    st.metric(
        label="Pasajeros Predichos (Cantidad) para la ciudad de Mendoza.",
        value=f"{int(st.session_state.prediccion_final):,}"
    )

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

    _, col_btn_center, _ = st.columns([1, 1, 1]) # 3 columnas, usamos la del medio
    with col_btn_center:
        if st.button("Hacer otra predicción", use_container_width=True):
            st.session_state.prediction_made = False
            st.rerun()