# --- app.py (Versión 5.6 - Hipótesis Ordenadas) ---

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

        df['Humedad_Alta'] = (df['Humedad_Media'] > 80).astype(int)


        df = df.sort_values(['Linea','Dia'])
        df['Cantidad_lag_1'] = df.groupby('Linea')['Cantidad'].shift(1)
        df['Cantidad_lag_7'] = df.groupby('Linea')['Cantidad'].shift(7)
        df['Cantidad_ma_7'] = df.groupby('Linea')['Cantidad'].transform(lambda x: x.rolling(7, min_periods=1).mean())

        for col in ['Cantidad_lag_1', 'Cantidad_lag_7', 'Cantidad_ma_7']:
            df[col] = df.groupby('Linea')[col].transform(lambda x: x.fillna(x.mean()))

        try:
            df = df.fillna(df.mean(numeric_only=True))
        except TypeError:
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df[cat_cols] = df[cat_cols].fillna('missing')

        return df
    except FileNotFoundError:
        return None

@st.cache_data
def get_viz_dataframes(df_raw):
    """
    Realiza los cálculos pesados para los gráficos ESTÁTICOS.
    """
    df = df_raw.copy()

    # 1. Datos para Gráfico Semanal
    df_semanal = df.set_index('Dia').resample('W')['Cantidad'].sum().reset_index()
    df_semanal['Anio'] = df_semanal['Dia'].dt.year
    anios = sorted(df['Dia'].dt.year.unique())

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

    return df_semanal, df_linea, df_dia, hist_data, anios

# --- 4. CARGAR DATOS Y MODELOS---
pipeline, promedios_data = load_models()
df_viz_raw = load_csv_data(CSV_PATH)

# Variables globales para predicción
if promedios_data and df_viz_raw is not None:
    promedios_df = promedios_data['promedios']
    media_global = promedios_data['media_global']
    promedios_lookup = promedios_df.set_index(['Linea', 'Dia_Semana'])
    LINEAS_EJEMPLO = sorted(df_viz_raw['Linea'].unique())
else:
    promedios_df, media_global, promedios_lookup = None, 0, None
    LINEAS_EJEMPLO = ["Error: Cargar Modelos"]

# --- 5. TÍTULO PRINCIPAL Y PESTAÑAS ---
st.title("🚌 Proyecto de Predicción de Pasajeros")

tab_info, tab_eval, tab_viz = st.tabs([
    "Sobre el Proyecto",
    "Evaluación del Modelo",
    "Visualizaciones"
])

# --- PESTAÑA 1: SOBRE EL PROYECTO ---
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

# --- PESTAÑA 2: EVALUACIÓN DEL MODELO ---
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
                    last_n_dates = df_viz_raw['Dia'].sort_values(ascending=False).unique()[:n_dias][::-1]
                    df_period = df_viz_raw[df_viz_raw['Dia'].isin(last_n_dates)].copy()

                    if linea_seleccionada == "Todas las líneas (Agregado)":
                        st.info("Mostrando la suma agregada de todas las líneas para los últimos N días.")
                        X_eval = df_period.copy()
                    else:
                        st.info(f"Mostrando la evaluación para: {linea_seleccionada}")
                        X_eval = df_period[df_period['Linea'] == linea_seleccionada].copy()
                        if X_eval.empty:
                            st.warning(f"No se encontraron datos para '{linea_seleccionada}' en los últimos {n_dias} días.")
                            st.stop()

                    for col in CAT_FEATURES:
                        X_eval[col] = X_eval[col].fillna('missing').astype(str)
                    numeric_cols = X_eval.select_dtypes(include=np.number).columns
                    X_eval[numeric_cols] = X_eval[numeric_cols].fillna(0)

                    prediccion_log = pipeline.predict(X_eval)
                    X_eval['Prediccion'] = np.expm1(prediccion_log)
                    X_eval['Valor Real'] = X_eval['Cantidad']

                    df_agg = X_eval.groupby('Dia')[['Valor Real', 'Prediccion']].sum().reset_index()
                    df_grafico = df_agg.rename(columns={'Prediccion': 'Predicción del Modelo'})

                    df_grafico['Fecha Real'] = df_grafico['Dia'].dt.strftime('%Y-%m-%d')
                    df_grafico['Dia (Nro)'] = np.arange(1, len(df_grafico) + 1)

                    df_melted = df_grafico.melt(
                        id_vars=['Dia (Nro)', 'Fecha Real'],
                        value_vars=['Valor Real', 'Predicción del Modelo'],
                        var_name='Tipo de Valor',
                        value_name='Cantidad_Melted'
                    )

                    st.subheader(f"Comparación de los Últimos {n_dias} Días ({linea_seleccionada})")

                    chart = alt.Chart(df_melted).mark_line(point=True).encode(
                        x=alt.X('Dia (Nro)', title=f'Últimos {n_dias} Días (en orden)', axis=alt.Axis(format='d')),
                        y=alt.Y('Cantidad_Melted', title='Cantidad de Pasajeros'),
                        color=alt.Color('Tipo de Valor', title="Valor:"),
                        tooltip=[
                            alt.Tooltip('Dia (Nro)', title="Día Nro."),
                            alt.Tooltip('Fecha Real'),
                            'Tipo de Valor',
                            alt.Tooltip('Cantidad_Melted', title="Cantidad", format=',.0f')
                        ]
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)

                    mae = mean_absolute_error(df_grafico['Valor Real'], df_grafico['Predicción del Modelo'])
                    st.metric(
                        label=f"Error Absoluto Medio (MAE) para este período",
                        value=f"{mae:,.2f} pasajeros"
                    )
                    st.info(f"El MAE indica que, en promedio, las predicciones del modelo para este período se desvían en {mae:,.2f} pasajeros del valor real.")

                except Exception as e:
                    st.error(f"Ocurrió un error al generar la predicción: {e}")
                    st.exception(e)

# --- PESTAÑA 3: VISUALIZACIONES ---
with tab_viz:
    st.header("Hallazgos y Visualizaciones")

    if df_viz_raw is None:
        st.error(f"Error: No se encontró el archivo en la ruta: {CSV_PATH}")
        st.info("Asegúrate de que el archivo CSV esté en la carpeta 'dev/' y subido a GitHub.")
    else:
        st.markdown(f"Exploración de los datos históricos (`{os.path.basename(CSV_PATH)}`)")

        with st.spinner("Procesando gráficos por primera vez..."):
            df_semanal, df_linea, df_dia, hist_data, anios = get_viz_dataframes(df_viz_raw)

        st.subheader("1. Serie Temporal de Pasajeros agrupado por semana")
        st.markdown("Se muestra la cantidad total de pasajeros por semana para ver la tendencia general, patrones estacionales y el impacto de eventos como la pandemia.")

        anios_opciones = ["Todos los Años"] + anios
        anio_seleccionado = st.selectbox("Filtrar por Año:", anios_opciones)

        if anio_seleccionado == "Todos los Años":
            df_semanal_filtrado = df_semanal
        else:
            df_semanal_filtrado = df_semanal[df_semanal['Anio'] == anio_seleccionado]

        chart_semanal = alt.Chart(df_semanal_filtrado).mark_line().encode(
            x=alt.X('Dia', title='Fecha'),
            y=alt.Y('Cantidad', title='Pasajeros Totales (por Semana)'),
            tooltip=['Dia', 'Cantidad']
        ).interactive()
        st.altair_chart(chart_semanal, use_container_width=True)

        st.divider()

        st.header("2. Exploración Interactiva por Línea")
        st.markdown("Usá el filtro para ver los patrones de líneas específicas. **Si no seleccionas ninguna, se mostrará el promedio de todas.**")

        lineas_seleccionadas = st.multiselect(
            "Selecciona una o más líneas para filtrar los gráficos:",
            options=LINEAS_EJEMPLO,
            default=[]
        )

        if not lineas_seleccionadas:
            df_filtrado = df_viz_raw
            titulo_filtro = "(Promedio de Todas las Líneas)"
        else:
            df_filtrado = df_viz_raw[df_viz_raw['Linea'].isin(lineas_seleccionadas)]
            if len(lineas_seleccionadas) > 3:
                titulo_filtro = f"({len(lineas_seleccionadas)} líneas seleccionadas)"
            else:
                titulo_filtro = f"({', '.join(lineas_seleccionadas)})"

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Pasajeros Promedio por Día {titulo_filtro}")

            df_dia_filtrado = df_filtrado.groupby('Dia_Semana')['Cantidad'].mean().reset_index()

            dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            base = alt.Chart(df_dia_filtrado).encode(
                y=alt.Y('Dia_Semana', sort=dias_orden, title='Día de la Semana'),
                x=alt.X('Cantidad', title='Pasajeros Promedio'),
                tooltip=['Dia_Semana', alt.Tooltip('Cantidad', format=',.0f')]
            )
            bars = base.mark_bar()
            text = base.mark_text(align='left', baseline='middle', dx=3).encode(
                text=alt.Text('Cantidad', format=',.0f'), color=alt.value('black')
            )
            chart_dia = (bars + text).interactive()
            st.altair_chart(chart_dia, use_container_width=True)

        with col2:
            st.subheader(f"Pasajeros Promedio por Temperatura {titulo_filtro}")


            df_filtrado['Temp_Redondeada'] = df_filtrado['Temp_media'].round()
            df_temp_filtrado = df_filtrado.groupby('Temp_Redondeada')['Cantidad'].mean().reset_index()

            chart_temp = alt.Chart(df_temp_filtrado).mark_bar().encode(
                x=alt.X('Temp_Redondeada:Q', title='Temperatura Media (°C)'), # ':Q' la trata como numérica
                y=alt.Y('Cantidad', title='Pasajeros Promedio'),
                tooltip=[alt.Tooltip('Temp_Redondeada', title="Temp. Media"), alt.Tooltip('Cantidad', format=',.0f')]
            ).interactive()
            st.altair_chart(chart_temp, use_container_width=True)

        st.divider()

        st.header("3. Gráficos Generales (Dataset Completo)")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Pasajeros Promedio por Línea")
            st.markdown("Promedio de pasajeros de todas las líneas a lo largo de todo el dataset.")
            chart_linea = alt.Chart(df_linea).mark_bar().encode(
                x=alt.X('Linea', sort='-y', title='Línea'),
                y=alt.Y('Cantidad', title='Pasajeros Promedio'),
                tooltip=['Linea', alt.Tooltip('Cantidad', format=',.0f')]
            ).interactive()
            st.altair_chart(chart_linea, use_container_width=True)

        with col4:
            st.subheader("Distribución de la variable Cantidad")
            st.markdown("Distribución general (sin filtrar) de la cantidad de pasajeros por registro.")
            hist_cantidad = alt.Chart(hist_data).mark_bar().encode(
                x=alt.X('Rango_Etiqueta', sort=None, title='Cantidad de Pasajeros (bins)'),
                y=alt.Y('Frecuencia', title='Frecuencia'),
                tooltip=['Rango_Etiqueta', 'Frecuencia']
            ).interactive()
            st.altair_chart(hist_cantidad, use_container_width=True)

        st.divider()

        st.header("4. Validación de Hipótesis")
        st.markdown("Validamos las hipótesis del proyecto analizando el promedio de pasajeros bajo diferentes condiciones.")

        @st.cache_data
        def crear_grafico_hipotesis(df, col_categorica, titulo, orden_x=None):
            """Función helper para crear los gráficos de hipótesis de forma consistente."""
            df_agg = df.groupby(col_categorica)['Cantidad'].mean().reset_index()

            if orden_x:
                eje_x = alt.X(col_categorica, title=titulo, sort=orden_x)
            else:
                 eje_x = alt.X(col_categorica, title=titulo, sort='-y')

            base = alt.Chart(df_agg).encode(
                x=eje_x,
                y=alt.Y('Cantidad', title='Pasajeros Promedio'),
                tooltip=[col_categorica, alt.Tooltip('Cantidad', format=',.0f')]
            )
            bars = base.mark_bar()
            text = base.mark_text(dy=-8).encode(
                text=alt.Text('Cantidad', format=',.0f'),
                color=alt.value('black')
            )
            return bars + text

        h_col1, h_col2, h_col3 = st.columns(3)

        with h_col1:
            st.markdown("**H1: Menos viajes en feriados**")
            df_viz_raw['H1_Feriado'] = df_viz_raw['Feriado'].map({0: 'Día No Feriado', 1: 'Día Feriado'})
            chart_h1 = crear_grafico_hipotesis(df_viz_raw, 'H1_Feriado', 'Tipo de Día')
            st.altair_chart(chart_h1, use_container_width=True)

        with h_col2:
            st.markdown("**H2: Caída en feriado + temp. extrema**")
            df_viz_raw['H2_Feriado_Extremo'] = df_viz_raw['Feriado_TempExtrema'].map({1: 'Feriado + Extremo', 0: 'Otro Día'})
            chart_h2 = crear_grafico_hipotesis(df_viz_raw, 'H2_Feriado_Extremo', 'Condición')
            st.altair_chart(chart_h2, use_container_width=True)

        with h_col3:
            st.markdown("**H3: Más viajes en la semana**")
            df_viz_raw['H3_Finde'] = df_viz_raw['Es_FinDeSemana'].map({0: 'Día de Semana', 1: 'Fin de Semana'})
            chart_h3 = crear_grafico_hipotesis(df_viz_raw, 'H3_Finde', 'Tipo de Día', orden_x=['Día de Semana', 'Fin de Semana'])
            st.altair_chart(chart_h3, use_container_width=True)

        h_col4, h_col5, h_col6 = st.columns(3)

        with h_col4:
            st.markdown("**H4: Caída en finde + mal clima**")
            df_viz_raw['H4_Adversa_Finde'] = df_viz_raw['Adversa_Finde'].map({1: 'Finde + Adverso', 0: 'Otro Día'})
            chart_h4 = crear_grafico_hipotesis(df_viz_raw, 'H4_Adversa_Finde', 'Condición')
            st.altair_chart(chart_h4, use_container_width=True)

        with h_col5:
            st.markdown("**H5: Menos viajes con lluvia/niebla**")
            df_viz_raw['H5_Adverso'] = df_viz_raw['Condicion_Adversa'].map({0: 'Día Normal', 1: 'Día Adverso'})
            chart_h5 = crear_grafico_hipotesis(df_viz_raw, 'H5_Adverso', 'Condición Climática', orden_x=['Día Normal', 'Día Adverso'])
            st.altair_chart(chart_h5, use_container_width=True)

        with h_col6:
            st.markdown("**H6: Menos viajes con humedad alta**")
            df_viz_raw['H6_Humedad'] = (df_viz_raw['Humedad_Media'] > 80).map({True: 'Húmedo (>80%)', False: 'Normal (<=80%)'})
            chart_h6 = crear_grafico_hipotesis(df_viz_raw, 'H6_Humedad', 'Nivel de Humedad')
            st.altair_chart(chart_h6, use_container_width=True)

        h_col7, _, _ = st.columns(3)
        with h_col7:
            st.markdown("**H7: Más viajes en temp. templada**")
            df_viz_raw['H7_Temp'] = df_viz_raw['Temp_Templada'].map({1: 'Templado (8-28°C)', 0: 'Extremo'})
            chart_h7 = crear_grafico_hipotesis(df_viz_raw, 'H7_Temp', 'Tipo de Temperatura', orden_x=['Templado (8-28°C)', 'Extremo'])
            st.altair_chart(chart_h7, use_container_width=True)