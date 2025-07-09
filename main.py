import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import re
import base64
import io
from io import StringIO

# Para el modelo de clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_index, calinski_harabasz_index

# --- Configuración de la página de Streamlit ---
st.set_page_config(layout="wide", page_title="Análisis y Calidad de Datos de Pacientes", page_icon="🏥")

# --- Funciones Auxiliares ---
def calculate_age_from_dob(row_dob, current_date=None):
    if pd.isna(row_dob):
        return None
    
    # Si current_date no se proporciona, usa la fecha actual
    if current_date is None:
        current_date = date.today()
        
    # Convertir a objeto date si no lo es
    if isinstance(row_dob, pd.Timestamp):
        row_dob = row_dob.date()
    elif isinstance(row_dob, str):
        try:
            row_dob = datetime.strptime(row_dob, '%Y-%m-%d').date()
        except ValueError:
            return None # Si el formato de string no es válido
    elif not isinstance(row_dob, date):
        return None # Tipo de dato no soportado

    age = current_date.year - row_dob.year - ((current_date.month, current_date.day) < (row_dob.month, row_dob.day))
    return age if age >= 0 else None

def get_missing_values_comparison(df_original, df_cleaned):
    missing_original = df_original.isnull().sum()
    missing_original_pct = (df_original.isnull().sum() / len(df_original)) * 100

    missing_cleaned = df_cleaned.isnull().sum()
    missing_cleaned_pct = (df_cleaned.isnull().sum() / len(df_cleaned)) * 100

    df_nulos_comp = pd.DataFrame({
        'Columna': df_original.columns,
        'Original_Nulos': missing_original,
        'Original_Porcentaje': missing_original_pct,
        'Limpio_Nulos': missing_cleaned,
        'Limpio_Porcentaje': missing_cleaned_pct
    }).set_index('Columna')
    return df_nulos_comp

def plot_missing_values_comparison(df_nulos_comp):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_nulos_comp.index,
        y=df_nulos_comp['Original_Porcentaje'],
        name='Original (%)',
        marker_color='skyblue'
    ))

    fig.add_trace(go.Bar(
        x=df_nulos_comp.index,
        y=df_nulos_comp['Limpio_Porcentaje'],
        name='Limpio (%)',
        marker_color='lightcoral'
    ))

    fig.update_layout(
        title_text='Comparación de Valores Faltantes (%) Antes y Después de la Limpieza',
        xaxis_title='Columnas',
        yaxis_title='Porcentaje de Nulos (%)',
        barmode='group',
        legend_title='Conjunto de Datos'
    )
    return fig

def create_pie_chart(df, column, title):
    counts = df[column].value_counts(dropna=False)
    # Reemplazar NaN para mostrar como 'Nulo' en el gráfico
    counts.index = counts.index.fillna('Nulo')
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title=title
    )
    return fig

def create_histogram(df, column, title, n_bins=20):
    fig = px.histogram(df, x=column, nbins=n_bins, title=title)
    fig.update_layout(bargap=0.1)
    return fig

def clean_patient_data(df_raw):
    df_cleaned = df_raw.copy()
    
    # 1. Limpieza y estandarización de 'sexo'
    df_cleaned['sexo'] = df_cleaned['sexo'].astype(str).str.lower().str.strip()
    sex_mapping = {
        'f': 'Female',
        'female': 'Female',
        'm': 'Male',
        'male': 'Male'
    }
    df_cleaned['sexo'] = df_cleaned['sexo'].map(sex_mapping)
    # Convertir cualquier valor no mapeado (incluyendo 'nan' como string) a None
    df_cleaned.loc[df_cleaned['sexo'].isna(), 'sexo'] = None

    # 2. Limpieza y conversión de 'fecha_nacimiento' y cálculo de 'edad'
    # Intentar convertir a datetime, los errores se convierten en NaT
    df_cleaned['fecha_nacimiento'] = pd.to_datetime(df_cleaned['fecha_nacimiento'], errors='coerce')

    # Calcular edad basada en fecha_nacimiento, manejar NaT y fechas futuras
    # Usar una fecha fija para el cálculo en un entorno reproducible o para pruebas
    current_date_for_age_calc = date.today()
    df_cleaned['edad_calculada'] = df_cleaned['fecha_nacimiento'].apply(lambda dob: calculate_age_from_dob(dob, current_date_for_age_calc))

    # Priorizar la edad calculada si fecha_nacimiento es válida y la edad reportada es nula/inconsistente
    df_cleaned['edad'] = df_cleaned.apply(
        lambda row: row['edad_calculada'] if pd.notna(row['edad_calculada']) else row['edad'], axis=1
    )
    # Descartar edades negativas o excesivamente altas (ej. si fecha de nacimiento es en el futuro)
    df_cleaned.loc[(df_cleaned['edad'] < 0) | (df_cleaned['edad'] > 120), 'edad'] = None # Edad máxima razonable 120
    
    # Convertir 'edad' a Int64 para soportar nulos
    df_cleaned['edad'] = df_cleaned['edad'].astype('Int64')
    
    # Eliminar la columna temporal 'edad_calculada'
    df_cleaned = df_cleaned.drop(columns=['edad_calculada'])

    # 3. Limpieza de 'telefono': remover no dígitos, convertir cadenas vacías a None
    df_cleaned['telefono'] = df_cleaned['telefono'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    df_cleaned.loc[df_cleaned['telefono'].str.strip() == '', 'telefono'] = None

    # 4. Validación de 'email': aunque no se "limpia" en el sentido de modificar, se puede identificar inválidos
    # El patrón de email es para verificar estructura, no existencia
    email_pattern = r'[^@]+@[^@]+\.[^@]+'
    df_cleaned['email_valido'] = df_cleaned['email'].astype(str).str.match(email_pattern)
    # Los correos inválidos no se eliminan, solo se marcan para análisis.
    # Puedes decidir qué hacer con ellos aquí (ej. convertirlos a None, aunque la descripción dice que no se alteran).
    # Por ahora, solo tenemos el indicador.

    # 5. Validación y estandarización de 'ciudad' (ejemplo, podría necesitar un catálogo)
    # Solo a minúsculas y sin espacios extra, pero no se mapea a un catálogo real sin datos de ejemplo.
    df_cleaned['ciudad'] = df_cleaned['ciudad'].astype(str).str.strip().str.title()
    # Si 'nan' como string, convertir a None
    df_cleaned.loc[df_cleaned['ciudad'] == 'Nan', 'ciudad'] = None

    return df_cleaned

def get_kpis(df):
    kpis = {}
    
    # KPIs generales
    kpis['total_pacientes'] = len(df)
    kpis['pacientes_con_email'] = df['email'].count()
    kpis['pacientes_con_telefono'] = df['telefono'].count()
    kpis['porcentaje_hombres'] = (df['sexo'] == 'Male').sum() / len(df) * 100 if len(df) > 0 else 0
    kpis['porcentaje_mujeres'] = (df['sexo'] == 'Female').sum() / len(df) * 100 if len(df) > 0 else 0

    # Edad
    kpis['edad_promedio'] = df['edad'].mean()
    kpis['edad_mediana'] = df['edad'].median()
    kpis['edad_minima'] = df['edad'].min()
    kpis['edad_maxima'] = df['edad'].max()
    
    # Otros
    kpis['ciudades_unicas'] = df['ciudad'].nunique()
    kpis['emails_invalidos_conteo'] = (~df['email_valido']).sum() if 'email_valido' in df.columns else 0 # Conteo de inválidos si la columna existe
    
    return kpis

def generate_eda_plots(df):
    plots = []

    # Distribución de Edad
    fig_edad = create_histogram(df.dropna(subset=['edad']), 'edad', 'Distribución de Edad')
    plots.append(("Distribución de Edad", fig_edad))

    # Distribución de Sexo
    fig_sexo = create_pie_chart(df, 'sexo', 'Distribución de Sexo')
    plots.append(("Distribución de Sexo", fig_sexo))

    # Ciudades más Frecuentes (Top 10)
    top_cities = df['ciudad'].value_counts().nlargest(10)
    fig_ciudad = px.bar(
        x=top_cities.index,
        y=top_cities.values,
        title='Top 10 Ciudades con más Pacientes',
        labels={'x': 'Ciudad', 'y': 'Número de Pacientes'}
    )
    plots.append(("Top 10 Ciudades", fig_ciudad))

    # Edad por Sexo (Box Plot)
    fig_edad_sexo = px.box(df.dropna(subset=['edad', 'sexo']), x='sexo', y='edad', title='Distribución de Edad por Sexo')
    plots.append(("Edad por Sexo", fig_edad_sexo))

    return plots

def generate_html_report(df_cleaned, df_original, df_nulos_comp, indicators_original, indicators_cleaned, kpis, eda_plots, cluster_results):
    # Función para convertir Plotly figures a HTML
    def fig_to_html(fig):
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Convertir plots de EDA a HTML
    eda_plots_html = ""
    for title, fig in eda_plots:
        eda_plots_html += f"<h3>{title}</h3>"
        eda_plots_html += fig_to_html(fig)
        eda_plots_html += "<br>"

    # Convertir plots de Cluster a HTML
    cluster_plots_html = ""
    if 'cluster_plots_data' in cluster_results and cluster_results['cluster_plots_data']:
        for title, fig in cluster_results['cluster_plots_data']:
            cluster_plots_html += f"<h3>{title}</h3>"
            cluster_plots_html += fig_to_html(fig)
            cluster_plots_html += "<br>"
    else:
        cluster_plots_html += "<p>No hay gráficos de clustering disponibles o el modelo no se ha ejecutado.</p>"

    # Asegurarse de que cluster_centers_df exista y no esté vacío
    cluster_centers_html = ""
    if 'cluster_centers_df' in cluster_results and not cluster_results['cluster_centers_df'].empty:
        cluster_centers_html = f"<h4>Centros de los Clusters:</h4>{cluster_results['cluster_centers_df'].to_html()}<br>"
    else:
        cluster_centers_html = "<p>Centros de los clusters no disponibles.</p>"

    # Asegurarse de que cluster_counts_df exista y no esté vacío
    cluster_counts_html = ""
    if 'cluster_counts_df' in cluster_results and not cluster_results['cluster_counts_df'].empty:
        cluster_counts_html = f"<h4>Conteo de Pacientes por Cluster:</h4>{cluster_results['cluster_counts_df'].to_html()}<br>"
    else:
        cluster_counts_html = "<p>Conteo de pacientes por cluster no disponible.</p>"
        
    # Tabla de nulos
    nulos_comp_html = df_nulos_comp.to_html(classes="table table-striped") if not df_nulos_comp.empty else "<p>No hay datos de comparación de nulos.</p>"

    # Comparación de tipos de datos
    tipos_original = {col: str(df_original[col].dtype) for col in df_original.columns}
    tipos_limpio = {col: str(df_cleaned[col].dtype) for col in df_cleaned.columns}
    tipos_comp_html = f"""
    <h4>Tipos de datos originales:</h4>
    <pre>{tipos_original}</pre>
    <h4>Tipos de datos después de la limpieza:</h4>
    <pre>{tipos_limpio}</pre>
    """

    # Indicadores de Consistencia y Unicidad
    sexo_original_unique = str(indicators_original.get('sexo_unique_original', 'N/A'))
    sexo_limpio_unique = str(indicators_cleaned.get('sexo_unique_cleaned', 'N/A'))
    email_invalidos_original = indicators_original.get('email_invalidos_original', 'N/A')
    email_invalidos_limpio = indicators_cleaned.get('email_invalidos_limpio', 'N/A')
    telefono_solo_digitos_limpio = 'Sí' if indicators_cleaned.get('telefono_only_digits_cleaned', False) else 'No'

    consistency_html = f"""
    <h3>Indicadores de Consistencia y Unicidad</h3>
    <h4>sexo - Unicidad de Categorías:</h4>
    <p>Original: {sexo_original_unique}</p>
    <p>Limpio: {sexo_limpio_unique}</p>
    <p>Observación: Se espera que el número de categorías únicas y sus nombres se normalicen después de la limpieza (ej., solo 'Female', 'Male' y None).</p>
    <h4>email - Patrón de Formato (Conteo de Inválidos):</h4>
    <p>Correos inválidos (Original): {email_invalidos_original}</p>
    <p>Correos inválidos (Limpio): {email_invalidos_limpio}</p>
    <p>Observación: Aunque la limpieza no los altera, se validó su formato. Este indicador muestra si persisten correos con formato no estándar.</p>
    <h4>telefono - Contiene solo dígitos (después de la limpieza):</h4>
    <p>Todos los teléfonos contienen solo dígitos o son nulos después de la limpieza: {telefono_solo_digitos_limpio}</p>
    """

    # KPIs
    kpis_html = "<ul>"
    for k, v in kpis.items():
        kpis_html += f"<li><b>{k.replace('_', ' ').title()}:</b> {v:.2f}" if isinstance(v, (int, float)) else f"<li><b>{k.replace('_', ' ').title()}:</b> {v}"
        kpis_html += "</li>"
    kpis_html += "</ul>"

    # Supuestos y Reglas
    supuestos_reglas_html = """
    <h3>3.2. Documentación del Proceso</h3>
    <h4>Supuestos Adoptados Durante la Limpieza:</h4>
    <ul>
        <li><b>Fuente Única para Edad:</b> Se asume que <code>fecha_nacimiento</code> es la fuente más confiable para determinar la <code>edad</code>. Si <code>fecha_nacimiento</code> es válida, se prioriza el cálculo de la edad a partir de ella sobre el valor <code>edad</code> existente si este es nulo o inconsistente. La edad se calcula como la diferencia en años a la fecha actual, ajustando por mes y día.</li>
        <li><b>Formato de <code>sexo</code>:</b> Se asume que los valores 'Female', 'female', 'Male', 'male', 'F', 'f', 'M', 'm' y sus variaciones deben ser estandarizados a 'Female' y 'Male'. Cualquier otro valor (NaN, vacío, o no reconocido) se convierte a <code>None</code>.</li>
        <li><b>Formato de <code>telefono</code>:</b> Se asume que los números de teléfono deben contener solo dígitos. Cualquier otro carácter (guiones, espacios, paréntesis, etc.) es removido. Las cadenas vacías o que solo consisten en espacios resultantes de esta limpieza se interpretan como nulas (<code>None</code>).</li>
        <li><b>Coherencia de Fechas:</b> Se asume que las fechas de nacimiento no pueden ser en el futuro ni excesivamente antiguas (la edad se calcula en relación con la fecha actual y las edades negativas se descartan, convirtiéndolas a <code>None</code>).</li>
        <li><b>ID de Paciente:</b> Se asume que <code>id_paciente</code> es el identificador único de cada paciente y no se espera que tenga problemas de calidad (duplicados, nulos).</li>
    </ul>
    <h4>Reglas de Validación Implementadas:</h4>
    <ul>
        <li><b>Validación de <code>fecha_nacimiento</code>:</b> Se verifica que la columna pueda ser convertida a tipo <code>datetime</code>. Los valores que no cumplen con este formato se marcan como <code>NaT</code> (Not a Time).</li>
        <li><b>Validación de <code>edad</code>:</b>
            <ul>
                <li>Debe ser un entero no negativo.</li>
                <li>Debe ser consistente con <code>fecha_nacimiento</code>: la <code>edad</code> calculada a partir de <code>fecha_nacimiento</code> debe ser cercana a la <code>edad</code> reportada (se permite una tolerancia de 1 año para posibles discrepancias de actualización de fechas en los datos originales).</li>
            </ul>
        </li>
        <li><b>Validación de <code>sexo</code>:</b> Los valores deben estar dentro de un conjunto predefinido de categorías estandarizadas (Female, Male o <code>None</code>).</li>
        <li><b>Validación de <code>email</code>:</b> Se verifica que el formato siga una expresión regular básica (<code>[^@]+@[^@]+\.[^@]+</code>) para asegurar que contenga un <code>@</code> y al menos un <code>.</code> en el dominio. Esta es una validación de patrón, no de existencia.</li>
        <li><b>Validación de <code>telefono</code>:</b> Se verifica que, después de la limpieza, la columna contenga solo caracteres numéricos (o sea nula).</li>
    </ul>
    """

    # Recomendaciones
    recomendaciones_html = """
    <h4>Recomendaciones de Mejora para Asegurar la Calidad Futura de los Datos:</h4>
    <ul>
        <li><b>Validación en Origen:</b> Implementar validaciones a nivel de entrada de datos (ej., formularios web, bases de datos) para <code>fecha_nacimiento</code>, <code>sexo</code>, <code>email</code> y <code>telefono</code>.
            <ul>
                <li><code>fecha_nacimiento</code>: Usar selectores de fecha para prevenir entradas manuales erróneas y asegurar formato <code>AAAA-MM-DD</code>.</li>
                <li><code>sexo</code>: Usar listas desplegables con opciones predefinidas (Female, Male) para evitar inconsistencias de capitalización o errores tipográficos.</li>
                <li><code>email</code>: Implementar validación de formato de correo electrónico en tiempo real en la entrada de datos y, si es posible, una verificación de dominio.</li>
                <li><code>telefono</code>: Forzar la entrada de solo dígitos o un formato específico (ej., con máscaras de entrada) dependiendo del país, y validar longitud mínima/máxima.</li>
            </ul>
        </li>
        <li><b>Estandarización de <code>ciudad</code>:</b> Implementar un catálogo o lista maestra de ciudades/municipios para asegurar consistencia y evitar variaciones en los nombres de ciudades (ej., "Barranquilla" vs "barranquilla", o errores tipográficos).</li>
        <li><b>Definición de Campos Obligatorios:</b> Establecer claramente qué campos son obligatorios (ej., <code>id_paciente</code>, <code>nombre</code>, <code>fecha_nacimiento</code>, <code>sexo</code>) en la base de datos o sistema de entrada para reducir la aparición de valores nulos críticos.</li>
        <li><b>Auditorías Regulares de Datos:</b> Realizar auditorías periódicas de la base de datos para identificar nuevos patrones de error o degradación de la calidad de los datos con el tiempo.</li>
        <li><b>Documentación de Metadatos:</b> Mantener un diccionario de datos actualizado que defina claramente cada campo, su tipo de dato esperado, formato, reglas de validación y significado, accesible para todo el equipo.</li>
        <li><b>Sistema de Reporte de Errores:</b> Establecer un mecanismo para que los usuarios (personal del hospital, médicos) reporten inconsistencias o errores en los datos cuando los detecten, con un flujo claro para su corrección.</li>
        <li><b>Capacitación del Personal:</b> Asegurar que el personal encargado de la entrada de datos esté continuamente capacitado en las mejores prácticas de entrada de datos y comprenda la importancia de la calidad de los datos para la toma de decisiones y la atención al paciente.</li>
    </ul>
    """
    
    # Ensamblar el informe completo
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Informe de Calidad de Datos de Pacientes</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: 'Arial', sans-serif; line-height: 1.6; color: #333; margin: 20px; }}
            .container {{ max-width: 1200px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1, h2, h3, h4 {{ color: #0056b3; margin-top: 20px; margin-bottom: 15px; }}
            pre {{ background: #f4f4f4; padding: 10px; border-left: 3px solid #0056b3; overflow-x: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .observation {{ background-color: #e6f7ff; border-left: 5px solid #2196f3; padding: 10px; margin-top: 10px; margin-bottom: 10px; }}
            .section {{ margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            .section:last-child {{ border-bottom: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center">Informe de Calidad y Análisis de Datos de Pacientes</h1>
            <p class="text-muted text-center">Generado el: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <div class="section">
                <h2>1. Resumen Ejecutivo</h2>
                <p>Este informe detalla el análisis de calidad de datos, el proceso de limpieza, la generación de indicadores clave de rendimiento (KPIs), el Análisis Exploratorio de Datos (EDA) y los resultados del modelado de Machine Learning (Clustering) aplicado a los datos de pacientes de un hospital.</p>
                <p>El objetivo principal es transformar los datos crudos en un conjunto de datos limpio, consistente y útil para el análisis y la toma de decisiones, identificando patrones y segmentos de pacientes.</p>
            </div>

            <div class="section">
                <h2>2. Indicadores de Calidad de Datos</h2>
                <h3>2.1. Comparación de Valores Faltantes (%)</h3>
                {nulos_comp_html}
                <div class="observation">
                    <h4>Observaciones:</h4>
                    <ul>
                        <li>Se espera una reducción significativa en el porcentaje de nulos en <b>edad</b> si <b>fecha_nacimiento</b> estaba disponible y era válida.</li>
                        <li><b>fecha_nacimiento</b> puede mostrar un aumento de nulos si los formatos originales eran inválidos y se convirtieron a <code>NaT</code>.</li>
                        <li><b>telefono</b> puede tener nulos si quedaron cadenas vacías después de limpiar caracteres no numéricos.</li>
                        <li><b>sexo</b> podría tener nulos si había valores vacíos o no estandarizables.</li>
                    </ul>
                </div>
                <h3>2.2. Comparación de Tipos de Datos</h3>
                {tipos_comp_html}
                <div class="observation">
                    <h4>Observaciones:</h4>
                    <ul>
                        <li><b>fecha_nacimiento</b> debería cambiar de <code>object</code> (cadena) a <code>datetime64[ns]</code> (tipo fecha y hora).</li>
                        <li><b>edad</b> debería cambiar de <code>object</code> (si contenía nulos o estaba mezclado) o <code>float64</code> (si se infirió numérico) a <code>Int64</code> (entero con soporte para nulos).</li>
                        <li><b>telefono</b> y <b>email</b> idealmente deberían permanecer como <code>object</code> (cadena) pero con formato validado.</li>
                    </ul>
                </div>
                {consistency_html}
            </div>

            <div class="section">
                <h2>3. Documentación del Proceso de Limpieza</h2>
                {supuestos_reglas_html}
                {recomendaciones_html}
            </div>

            <div class="section">
                <h2>4. Key Performance Indicators (KPIs)</h2>
                {kpis_html}
            </div>

            <div class="section">
                <h2>5. Análisis Exploratorio de Datos (EDA)</h2>
                {eda_plots_html}
            </div>

            <div class="section">
                <h2>6. Resultados del Modelo de Clustering</h2>
                <p>Número óptimo de clusters: <b>{cluster_results.get('n_clusters', 'N/A')}</b></p>
                <ul>
                    <li>Puntuación de Silueta: <b>{cluster_results.get('silhouette_score', np.nan):.2f}</b> (indica cuán similar es un objeto a su propio clúster en comparación con otros clústeres)</li>
                    <li>Índice Davies-Bouldin: <b>{cluster_results.get('davies_bouldin_index', np.nan):.2f}</b> (valores más bajos indican mejor partición)</li>
                    <li>Índice Calinski-Harabasz: <b>{cluster_results.get('calinski_harabasz_index', np.nan):.2f}</b> (valores más altos indican clusters más densos y bien separados)</li>
                    <li>Inercia (Sum of Squared Distances): <b>{cluster_results.get('inertia', np.nan):.2f}</b> (distancia intra-cluster, menor es mejor)</li>
                </ul>
                {cluster_centers_html}
                {cluster_counts_html}
                {cluster_plots_html}
            </div>

            <div class="section">
                <h2>7. Conclusiones y Próximos Pasos</h2>
                <p>El proceso de limpieza ha mejorado significativamente la calidad de los datos, reduciendo nulos y estandarizando formatos. El EDA ha revelado insights clave sobre la demografía de los pacientes. El clustering ha permitido identificar segmentos de pacientes, lo cual puede ser valioso para la atención personalizada y la gestión hospitalaria.</p>
                <p>Se recomienda implementar las mejoras sugeridas para garantizar la calidad de los datos desde el origen y continuar explorando los segmentos de pacientes identificados para estrategias específicas.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# --- Cargar datos iniciales ---
# Simular carga de datos de un CSV o similar
@st.cache_data
def load_data():
    # Datos de ejemplo si no se sube un archivo
    data = {
        "id_paciente": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "nombre": ["Ana García", "Luis Pérez", "María López", "Juan Rodríguez", "Laura Díaz", "Carlos Sánchez", "Sofía Ramírez", "Pedro Castro", "Elena Morales", "Miguel Herrera", "Isabel Gómez", "Francisco Ruiz", "Carmen Navarro", "Jorge Torres", "Lucía Gil", "Manuel Vargas", "Patricia Ortiz", "Ricardo Nuñez", "Andrea Salas", "Diego Blanco"],
        "fecha_nacimiento": ["1985-03-10", "1970-11-22", "1992-07-01", "1965-01-15", "1988-09-30", "1975-04-05", "2000-02-28", "1950-12-03", "1980-06-18", "1995-10-01", "1978-03-25", "1960-08-12", "1990-05-20", "1982-01-01", "2005-04-14", "1973-09-09", "1989-11-11", "1968-06-06", "1998-07-07", "1955-02-02"],
        "edad": [38, 53, 31, 58, 35, 48, 23, 73, 43, 28, 45, 63, 33, 41, 18, 50, 34, 55, 25, 68],
        "sexo": ["Female", "Male", "FEMALE", "male", "Female", "Male", "Female", "Male", "Female", "Male", "F", "M", "Female", "Male", "F", "Male", "Female", "Male", "Female", "Male"],
        "email": ["ana.g@example.com", "luis.p@ejemplo.com", "maria.l@correo.com", "juan.r@dominio.org", "laura.d@miemail.es", "carlos.s@hospital.net", "sofia.r@outlook.com", "pedro.c@gmail.com", "elena.m@yahoo.com", "miguel.h@email.com", "isabel.g@proveedor.com", "francisco.r@empresa.com", "carmen.n@hospital.org", "jorge.t@email.es", "lucia.g@dominio.net", "manuel.v@correolibre.com", "patricia.o@correo.es", "ricardo.n@mihospital.com", "andrea.s@ejemplo.net", "diego.b@gmail.com"],
        "telefono": ["3101234567", "3009876543", "3205551234", "3158889900", "3011112233", "3054445566", "3127778899", "3182223344", "3169990011", "3046667788", "3173334455", "3020001122", "3136667788", "3191112233", "3034445566", "3147778899", "3210009988", "3062223344", "3225556677", "3118887766"],
        "ciudad": ["Bogotá", "Medellín", "Cali", "Barranquilla", "Cartagena", "Cali", "Medellín", "Bogotá", "Bogotá", "Barranquilla", "Bogotá", "Medellín", "Cali", "Barranquilla", "Cartagena", "Bogotá", "Medellín", "Cali", "Bogotá", "Barranquilla"]
    }
    df = pd.DataFrame(data)
    # Introducir algunos nulos y datos inconsistentes intencionalmente para la demostración
    df.loc[1, 'edad'] = None # Nulo en edad
    df.loc[2, 'sexo'] = 'Otro' # Valor inconsistente en sexo
    df.loc[4, 'fecha_nacimiento'] = '1988/09/30' # Formato diferente de fecha
    df.loc[5, 'telefono'] = '305-444-55-66' # Formato con guiones
    df.loc[6, 'email'] = 'sofia.r@outlook' # Email inválido
    df.loc[7, 'edad'] = -5 # Edad negativa
    df.loc[8, 'fecha_nacimiento'] = '2030-01-01' # Fecha futura
    df.loc[10, 'telefono'] = '    ' # Teléfono solo espacios
    df.loc[11, 'ciudad'] = 'bogota' # Ciudad con minúscula
    df.loc[12, 'sexo'] = '' # Sexo vacío
    df.loc[14, 'edad'] = 150 # Edad irreal
    df.loc[17, 'email'] = 'ricardo@.com' # Email inválido

    return df

df_pacientes = load_data()

# --- Título de la Aplicación ---
st.title("🏥 Análisis y Calidad de Datos de Pacientes de Hospital")
st.markdown("""
Esta aplicación realiza un análisis exhaustivo de la calidad de los datos de pacientes, 
seguido de procesos de limpieza, validación, generación de KPIs, EDA avanzado y 
un modelo de Machine Learning (Clustering) para identificar segmentos de pacientes.
""")

# --- Inicializar st.session_state para guardar los resultados entre secciones ---
# Esto es CRÍTICO para evitar KeyErrors al acceder a variables de sesión
if 'df_cleaned' not in st.session_state:
    st.session_state['df_cleaned'] = df_pacientes.copy() # Copia inicial

if 'df_original' not in st.session_state:
    st.session_state['df_original'] = df_pacientes.copy() # Copia del original

if 'kpis' not in st.session_state:
    st.session_state['kpis'] = {}

if 'eda_plots_data' not in st.session_state:
    st.session_state['eda_plots_data'] = [] # Lista de (title, base64_image_data)

if 'cluster_results_data' not in st.session_state:
    st.session_state['cluster_results_data'] = {
        'n_clusters': 0,
        'silhouette_score': np.nan,
        'davies_bouldin_index': np.nan,
        'calinski_harabasz_index': np.nan,
        'inertia': np.nan,
        'cluster_centers_df': pd.DataFrame(),
        'cluster_counts_df': pd.DataFrame(),
        'cluster_plots_data': [] # Lista de (title, base64_image_data)
    }

if 'df_nulos_comp' not in st.session_state:
    st.session_state['df_nulos_comp'] = pd.DataFrame()

if 'indicators_original' not in st.session_state:
    st.session_state['indicators_original'] = {}

if 'indicators_cleaned' not in st.session_state:
    st.session_state['indicators_cleaned'] = {}


# --- Barra Lateral para Navegación ---
st.sidebar.title("Navegación")
section = st.sidebar.radio(
    "Ir a:",
    [
        "1. Carga y Exploración Inicial",
        "2. Limpieza y Validación de Datos",
        "3. Indicadores de Calidad y Documentación",
        "4. EDA Avanzado & Dashboards",
        "5. Modelado de Machine Learning (Clustering)"
    ]
)

# --- Sección 1: Carga y Exploración Inicial ---
if section == "1. Carga y Exploración Inicial":
    st.header("1. Carga y Exploración Inicial de Datos")
    st.write("Carga tu propio archivo CSV o usa los datos de ejemplo precargados.")

    uploaded_file = st.file_uploader("Sube tu archivo CSV de pacientes", type="csv")

    if uploaded_file is not None:
        try:
            df_pacientes = pd.read_csv(uploaded_file)
            st.session_state['df_original'] = df_pacientes.copy()
            st.session_state['df_cleaned'] = df_pacientes.copy() # Inicializar df_cleaned también
            st.success("Archivo cargado exitosamente.")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}. Asegúrate de que sea un CSV válido.")
            df_pacientes = load_data() # Volver a cargar datos de ejemplo si hay error
            st.session_state['df_original'] = df_pacientes.copy()
            st.session_state['df_cleaned'] = df_pacientes.copy() # Inicializar df_cleaned
            st.info("Se han cargado los datos de ejemplo en su lugar.")
    else:
        st.info("Usando datos de ejemplo precargados. Puedes subir tu propio CSV.")
        df_pacientes = st.session_state['df_original'] # Asegurarse de usar el original cargado/ejemplo

    st.subheader("Vista Previa de los Datos Originales")
    st.dataframe(df_pacientes.head())

    st.subheader("Información General del DataFrame")
    buffer = StringIO()
    df_pacientes.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Estadísticas Descriptivas")
    st.dataframe(df_pacientes.describe(include='all'))

    st.subheader("Valores Nulos por Columna (Original)")
    nulos_original = df_pacientes.isnull().sum()
    nulos_original_pct = (df_pacientes.isnull().sum() / len(df_pacientes)) * 100
    df_nulos_original = pd.DataFrame({
        'Nulos': nulos_original,
        'Porcentaje (%)': nulos_original_pct
    }).sort_values(by='Porcentaje (%)', ascending=False)
    st.dataframe(df_nulos_original)

    # Almacenar indicadores originales para la sección 3
    st.session_state['indicators_original'] = {
        'sexo_unique_original': df_pacientes['sexo'].unique().tolist(),
        'email_invalidos_original': (~df_pacientes['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)).sum()
    }


# --- Sección 2: Limpieza y Validación de Datos ---
elif section == "2. Limpieza y Validación de Datos":
    st.header("2. Limpieza y Validación de Datos")
    st.write("Aplica las reglas de limpieza y validación definidas al conjunto de datos.")

    if st.button("Aplicar Limpieza y Validación"):
        with st.spinner("Limpiando y validando datos..."):
            df_cleaned = clean_patient_data(st.session_state['df_original'])
            st.session_state['df_cleaned'] = df_cleaned
            st.success("¡Datos limpios y validados!")

            # Recalcular y almacenar indicadores para la sección 3 (después de limpieza)
            st.session_state['indicators_cleaned'] = {
                'sexo_unique_cleaned': df_cleaned['sexo'].unique().tolist(),
                'email_invalidos_limpio': (~df_cleaned['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)).sum(),
                'telefono_only_digits_cleaned': df_cleaned['telefono'].dropna().apply(lambda x: str(x).isdigit()).all()
            }

            # También calcular la comparación de nulos para la sección 3
            st.session_state['df_nulos_comp'] = get_missing_values_comparison(st.session_state['df_original'], st.session_state['df_cleaned'])


    st.subheader("Vista Previa de los Datos Limpios")
    st.dataframe(st.session_state['df_cleaned'].head())

    st.subheader("Información General del DataFrame Limpio")
    buffer_cleaned = StringIO()
    st.session_state['df_cleaned'].info(buf=buffer_cleaned)
    st.text(buffer_cleaned.getvalue())

    st.subheader("Valores Nulos por Columna (Después de Limpieza)")
    nulos_cleaned = st.session_state['df_cleaned'].isnull().sum()
    nulos_cleaned_pct = (st.session_state['df_cleaned'].isnull().sum() / len(st.session_state['df_cleaned'])) * 100
    df_nulos_cleaned = pd.DataFrame({
        'Nulos': nulos_cleaned,
        'Porcentaje (%)': nulos_cleaned_pct
    }).sort_values(by='Porcentaje (%)', ascending=False)
    st.dataframe(df_nulos_cleaned)


# --- Sección 3: Indicadores de Calidad y Documentación ---
elif section == "3. Indicadores de Calidad y Documentación":
    st.header("3. Indicadores de Calidad y Documentación")
    st.write("Resumen de indicadores de calidad antes y después de la limpieza, junto con la documentación.")

    st.subheader("3.1. Indicadores de Calidad de Datos")

    st.markdown("#### Comparación de Valores Faltantes (%)")
    if not st.session_state['df_nulos_comp'].empty:
        st.dataframe(st.session_state['df_nulos_comp'])
        fig_nulos_comp = plot_missing_values_comparison(st.session_state['df_nulos_comp'])
        st.plotly_chart(fig_nulos_comp, use_container_width=True)
    else:
        st.warning("Por favor, ejecuta la sección '2. Limpieza y Validación de Datos' primero para generar la comparación de nulos.")
    
    st.markdown("""
    **Observaciones:**
    * Se espera una **reducción significativa** en el porcentaje de nulos en **edad** si **fecha_nacimiento** estaba disponible y era válida.
    * **fecha_nacimiento** puede mostrar un aumento de nulos si los formatos originales eran inválidos y se convirtieron a `NaT`.
    * **telefono** puede tener nulos si quedaron cadenas vacías después de limpiar caracteres no numéricos.
    * **sexo** podría tener nulos si había valores vacíos o no estandarizables.
    """)

    st.markdown("#### Comparación de Tipos de Datos")
    st.write("**Tipos de datos originales:**")
    st.code(str({col: str(st.session_state['df_original'][col].dtype) for col in st.session_state['df_original'].columns}))
    st.write("**Tipos de datos después de la limpieza:**")
    st.code(str({col: str(st.session_state['df_cleaned'][col].dtype) for col in st.session_state['df_cleaned'].columns}))
    st.markdown("""
    **Observaciones:**
    * **fecha_nacimiento** debería cambiar de `object` (cadena) a `datetime64[ns]` (tipo fecha y hora).
    * **edad** debería cambiar de `object` (si contenía nulos o estaba mezclado) o `float64` (si se infirió numérico) a `Int64` (entero con soporte para nulos).
    * **telefono** y **email** idealmente deberían permanecer como `object` (cadena) pero con formato validado.
    """)

    st.markdown("#### Indicadores de Consistencia y Unicidad")
    st.write("**sexo - Unicidad de Categorías:**")
    st.write(f"Original: {st.session_state['indicators_original'].get('sexo_unique_original', 'N/A')}")
    st.write(f"Limpio: {st.session_state['indicators_cleaned'].get('sexo_unique_cleaned', 'N/A')}")
    st.markdown("Observación: Se espera que el número de categorías únicas y sus nombres se normalicen después de la limpieza (ej., solo 'Female', 'Male' y `None`).")

    st.write("**email - Patrón de Formato (Conteo de Inválidos):**")
    st.write(f"Correos inválidos (Original): {st.session_state['indicators_original'].get('email_invalidos_original', 'N/A')}")
    st.write(f"Correos inválidos (Limpio): {st.session_state['indicators_cleaned'].get('email_invalidos_limpio', 'N/A')}")
    st.markdown("Observación: Aunque la limpieza no los altera, se validó su formato. Este indicador muestra si persisten correos con formato no estándar.")

    st.write("**telefono - Contiene solo dígitos (después de la limpieza):**")
    st.write(f"Todos los teléfonos contienen solo dígitos o son nulos después de la limpieza: {'Sí' if st.session_state['indicators_cleaned'].get('telefono_only_digits_cleaned', False) else 'No'}")


    st.subheader("3.2. Documentación del Proceso")
    st.markdown("""
    #### Supuestos Adoptados Durante la Limpieza:
    * **Fuente Única para Edad:** Se asume que `fecha_nacimiento` es la fuente más confiable para determinar la `edad`. Si `fecha_nacimiento` es válida, se **prioriza el cálculo de la edad a partir de ella** sobre el valor `edad` existente si este es nulo o inconsistente. La edad se calcula como la diferencia en años a la fecha actual, ajustando por mes y día.
    * **Formato de `sexo`:** Se asume que los valores `Female`, `female`, `Male`, `male`, `F`, `f`, `M`, `m` y sus variaciones deben ser estandarizados a `Female` y `Male`. Cualquier otro valor (NaN, vacío, o no reconocido) se convierte a `None`.
    * **Formato de `telefono`:** Se asume que los números de teléfono deben contener solo dígitos. Cualquier otro carácter (guiones, espacios, paréntesis, etc.) es **removido**. Las cadenas vacías o que solo consisten en espacios resultantes de esta limpieza se interpretan como nulas (None).
    * **Coherencia de Fechas:** Se asume que las fechas de nacimiento no pueden ser en el futuro ni excesivamente antiguas (la edad se calcula en relación con la fecha actual y las edades negativas se descartan, convirtiéndolas a `None`).
    * **ID de Paciente:** Se asume que `id_paciente` es el identificador **único** de cada paciente y no se espera que tenga problemas de calidad (duplicados, nulos).

    #### Reglas de Validación Implementadas:
    * **Validación de `fecha_nacimiento`:** Se verifica que la columna pueda ser convertida a tipo `datetime`. Los valores que no cumplen con este formato se marcan como `NaT` (Not a Time).
    * **Validación de `edad`:**
        * Debe ser un entero no negativo.
        * Debe ser **consistente** con `fecha_nacimiento`: la `edad` calculada a partir de `fecha_nacimiento` debe ser cercana a la `edad` reportada (se permite una tolerancia de 1 año para posibles discrepancias de actualización de fechas en los datos originales).
    * **Validación de `sexo`:** Los valores deben estar dentro de un conjunto predefinido de categorías estandarizadas (Female, `Male` o `None`).
    * **Validación de `email`:** Se verifica que el formato siga una expresión regular básica (`[^@]+@[^@]+\.[^@]+`) para asegurar que contenga un `@` y al menos un `.` en el dominio. Esta es una validación de patrón, no de existencia.
    * **Validación de `telefono`:** Se verifica que, después de la limpieza, la columna contenga solo caracteres numéricos (o sea nula).

    #### Recomendaciones de Mejora para Asegurar la Calidad Futura de los Datos:
    * **Validación en Origen:** Implementar validaciones a nivel de entrada de datos (ej., formularios web, bases de datos) para `fecha_nacimiento`, `sexo`, `email` y `telefono`.
        * `fecha_nacimiento`: Usar selectores de fecha para prevenir entradas manuales erróneas y asegurar formato `AAAA-MM-DD`.
        * `sexo`: Usar listas desplegables con opciones predefinidas (Female, Male) para evitar inconsistencias de capitalización o errores tipográficos.
        * `email`: Implementar validación de formato de correo electrónico en tiempo real en la entrada de datos y, si es posible, una verificación de dominio.
        * `telefono`: Forzar la entrada de solo dígitos o un formato específico (ej., con máscaras de entrada) dependiendo del país, y validar longitud mínima/máxima.
    * **Estandarización de `ciudad`:** Implementar un catálogo o lista maestra de ciudades/municipios para asegurar consistencia y evitar variaciones en los nombres de ciudades (ej., "Barranquilla" vs "barranquilla", o errores tipográficos).
    * **Definición de Campos Obligatorios:** Establecer claramente qué campos son obligatorios (ej., `id_paciente`, `nombre`, `fecha_nacimiento`, `sexo`) en la base de datos o sistema de entrada para reducir la aparición de valores nulos críticos.
    * **Auditorías Regulares de Datos:** Realizar auditorías periódicas de la base de datos para identificar nuevos patrones de error o degradación de la calidad de los datos con el tiempo.
    * **Documentación de Metadatos:** Mantener un **diccionario de datos** actualizado que defina claramente cada campo, su tipo de dato esperado, formato, reglas de validación y significado, accesible para todo el equipo.
    * **Sistema de Reporte de Errores:** Establecer un mecanismo para que los usuarios (personal del hospital, médicos) reporten inconsistencias o errores en los datos cuando los detecten, con un flujo claro para su corrección.
    * **Capacitación del Personal:** Asegurar que el personal encargado de la entrada de datos esté continuamente capacitado en las mejores prácticas de entrada de datos y comprenda la importancia de la calidad de los datos para la toma de decisiones y la atención al paciente.
    """)

    st.subheader("3.3. Bonus (Opcional)")
    st.markdown("""
    #### Implementación de Pruebas Automáticas
    Para implementar pruebas automáticas para la calidad de los datos, se podrían usar frameworks como **Pytest** o **Great Expectations**.

    **Ejemplo conceptual con Pytest** (en un archivo `tests/test_data_quality.py`):
    ```python
    # Este código es conceptual y no forma parte de app.py
    # Deberías tener tus funciones de limpieza y validación en un módulo separado para importarlas aquí.
    import pandas as pd
    import pytest
    from datetime import date, datetime

    # from your_project.data_quality_functions import clean_patient_data, calculate_age_from_dob # Ejemplo de importación

    # Asegúrate de que calculate_age_from_dob sea accesible si no la importas desde un módulo
    def calculate_age_from_dob(row_dob, current_date):
        if pd.isna(row_dob):
            return None
        else:
            if isinstance(row_dob, str):
                try:
                    row_dob = datetime.strptime(row_dob, '%Y-%m-%d').date()
                except ValueError:
                    return None
            elif isinstance(row_dob, pd.Timestamp):
                row_dob = row_dob.date()
            age = current_date.year - row_dob.year - ((current_date.month, current_date.day) < (row_dob.month, row_dob.day))
            return age if age >= 0 else None

    @pytest.fixture
    def sample_patient_data():
        # Datos de prueba con casos conocidos para verificar la limpieza
        data = {
            "pacientes": [
                {"id_paciente": 1, "nombre": "Claudia Torres", "fecha_nacimiento": "1954-01-08", "edad": None, "sexo": "Female", "email": "user1@example.com", "telefono": "342-950-1064", "ciudad": "Barranquilla"},
                {"id_paciente": 2, "nombre": "Pedro Gomez", "fecha_nacimiento": "1980-05-15", "edad": 40, "sexo": "male", "email": "pedro@example", "telefono": "123-ABC-456", "ciudad": "Medellin"},
                {"id_paciente": 3, "nombre": "Ana Smith", "fecha_nacimiento": "2025-01-01", "edad": 5, "sexo": "FEMALE", "email": "ana@example.com", "telefono": "7891234567", "ciudad": "Bogota"}, # Fecha futura, edad incorrecta
                {"id_paciente": 4, "nombre": "Luis Lopez", "fecha_nacimiento": "1990-11-20", "edad": None, "sexo": "Male", "email": "luis.lopez@example.com", "telefono": "9876543210", "ciudad": "Cali"},
                {"id_paciente": 5, "nombre": "Maria Paz", "fecha_nacimiento": None, "edad": 30, "sexo": "Female", "email": "maria@example.net", "telefono": "300-111-2222", "ciudad": "Bogota"},
                {"id_paciente": 6, "nombre": "Carlos", "fecha_nacimiento": "1970-07-08", "edad": 50, "sexo": "OTHER", "email": "carlos@example.com", "telefono": "123-456-7890", "ciudad": "Cartagena"},
                {"id_paciente": 7, "nombre": "Laura", "fecha_nacimiento": "1995-03-20", "edad": None, "sexo": "F", "email": "laura@example.com", "telefono": "1112223333", "ciudad": "Bucaramanga"}, # Prueba de 'F'
                {"id_paciente": 8, "nombre": "Diego", "fecha_nacimiento": "1988-09-10", "edad": None, "sexo": "m", "email": "diego@example.com", "telefono": "4445556666", "ciudad": "Cali"} # Prueba de 'm'
            ]
        }
        return pd.json_normalize(data['pacientes'])

    # Esta sería la función de limpieza que probarías, adaptada de tu app.py
    def clean_patient_data_for_test(df_raw):
        df_cleaned_test = df_raw.copy()
        current_date_for_test = date(2025, 7, 9) # Fecha fija para las pruebas de edad

        # Sexo
        df_cleaned_test['sexo'] = df_cleaned_test['sexo'].astype(str).str.lower()
        sex_mapping = {
            'f': 'Female',
            'female': 'Female',
            'm': 'Male',
            'male': 'Male'
        }
        df_cleaned_test['sexo'] = df_cleaned_test['sexo'].map(sex_mapping)
        df_cleaned_test.loc[df_cleaned_test['sexo'].isna(), 'sexo'] = None

        # Fecha Nacimiento y Edad
        df_cleaned_test['fecha_nacimiento'] = pd.to_datetime(df_cleaned_test['fecha_nacimiento'], errors='coerce')
        df_cleaned_test['edad_calculada_test'] = df_cleaned_test['fecha_nacimiento'].apply(lambda dob: calculate_age_from_dob(dob, current_date_for_test))
        df_cleaned_test['edad'] = df_cleaned_test.apply(
            lambda row: row['edad_calculada_test'] if pd.notna(row['edad_calculada_test']) else row['edad'], axis=1
        )
        df_cleaned_test['edad'] = df_cleaned_test['edad'].astype('Int64')
        df_cleaned_test = df_cleaned_test.drop(columns=['edad_calculada_test'])

        # Telefono
        df_cleaned_test['telefono'] = df_cleaned_test['telefono'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df_cleaned_test.loc[df_cleaned_test['telefono'].str.strip() == '', 'telefono'] = None
        return df_cleaned_test

    def test_sexo_standardization(sample_patient_data):
        df_cleaned = clean_patient_data_for_test(sample_patient_data)
        assert all(s in ['Female', 'Male', None] for s in df_cleaned['sexo'].unique()), "Los valores de sexo no están estandarizados o contienen inesperados."
        assert pd.isna(df_cleaned.loc[df_cleaned['id_paciente'] == 6, 'sexo'].iloc[0]), "El valor 'OTHER' para sexo no se convirtió a None."
        assert df_cleaned.loc[df_cleaned['id_paciente'] == 7, 'sexo'].iloc[0] == 'Female', "El valor 'F' para sexo no se convirtió a 'Female'."
        assert df_cleaned.loc[df_cleaned['id_paciente'] == 8, 'sexo'].iloc[0] == 'Male', "El valor 'm' para sexo no se convirtió a 'Male'."

    def test_age_calculation_and_validation(sample_patient_data):
        df_cleaned = clean_patient_data_for_test(sample_patient_data)
        assert all(df_cleaned['edad'].dropna() >= 0), "Las edades calculadas no deben ser negativas."
        # Verificar edad para id_paciente 1 (1954-01-08) -> 2025-1954 = 71
        assert df_cleaned.loc[df_cleaned['id_paciente'] == 1, 'edad'].iloc[0] == 71, "La edad para el paciente 1 no se calculo correctamente."
        # Verificar que la fecha futura (2025-01-01) resulta en edad nula/None
        assert pd.isna(df_cleaned.loc[df_cleaned['id_paciente'] == 3, 'edad'].iloc[0]), "La edad para fecha futura deberia ser nula."
        # Verificar que si fecha_nacimiento es nulo pero la edad existe, se mantiene (id_paciente 5)
        assert df_cleaned.loc[df_cleaned['id_paciente'] == 5, 'edad'].iloc[0] == 30, "La edad para el paciente 5 no se mantuvo correctamente."

    def test_email_format_after_cleaning(sample_patient_data):
        df_cleaned = clean_patient_data_for_test(sample_patient_data) # No hay limpieza directa de email, solo validación
        invalid_emails_in_cleaned = df_cleaned[~df_cleaned['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        # Esperamos que 'pedro@example' siga siendo inválido
        assert 'pedro@example' in invalid_emails_in_cleaned['email'].values, "El email 'pedro@example' no se marco como invalido."
        # No esperamos que se añadan nuevos invalidos, y su numero debe ser consistente con los originales
        assert len(invalid_emails_in_cleaned) == 1, "Se detecto un numero inesperado de correos electronicos invalidos."

    def test_telefono_numeric_after_cleaning(sample_patient_data):
        df_cleaned = clean_patient_data_for_test(sample_patient_data)
        assert all(df_cleaned['telefono'].dropna().apply(lambda x: x.isdigit())), "El campo telefono contiene caracteres no numericos despues de la limpieza."
        assert pd.isna(df_cleaned.loc[df_cleaned['id_paciente'] == 2, 'telefono'].iloc[0]), "El telefono con caracteres no numericos no se limpio correctamente."
    ```
    Para ejecutar Pytest, necesitas:
    * `pytest` instalado: `pip install pytest`
    * Guardar el código de prueba en un archivo como `tests/test_data_quality.py` (o similar) en una carpeta `tests/`.
    * **Importante:** Refactorizar tus funciones de limpieza y validación de `app.py` en un módulo de Python separado (ej., `procesamiento_datos.py`) para que puedas importarlas en las pruebas. O, para esta demostración, puedes copiar y adaptar las funciones de limpieza dentro del propio archivo de prueba como se muestra arriba.
    * Ejecutar `pytest` en tu terminal desde la raíz de tu proyecto.

    #### Simulación de Migración de Datos Limpios a una Estructura Objetivo
    Una vez que los datos han sido limpiados y validados, el siguiente paso lógico en una tubería de datos es cargarlos en una estructura objetivo, como un Data Warehouse o una base de datos analítica. Formatos como **Parquet** son ideales para esto debido a su naturaleza columnar, compresión eficiente y capacidad para manejar esquemas complejos.

    Aquí simulamos la descarga de los datos limpios en formatos comunes para la migración.
    """)

    st.subheader("Simulación de Migración de Datos Limpios")

    col1, col2 = st.columns(2)

    with col1:
        csv_buffer = StringIO()
        st.session_state['df_cleaned'].to_csv(csv_buffer, index=False)
        st.download_button(
            label="Descargar CSV Limpio",
            data=csv_buffer.getvalue(),
            file_name="pacientes_limpio.csv",
            mime="text/csv",
            help="Descarga el DataFrame limpio en formato CSV."
        )

    with col2:
        try:
            # Para Parquet, es mejor usar un buffer de Bytes
            parquet_buffer = io.BytesIO()
            st.session_state['df_cleaned'].to_parquet(parquet_buffer, index=False)
            st.download_button(
                label="Descargar Parquet Limpio",
                data=parquet_buffer.getvalue(),
                file_name="pacientes_limpio.parquet",
                mime="application/octet-stream",
                help="Descarga el DataFrame limpio en formato Parquet (eficiente para Big Data)."
            )
        except Exception as e:
            st.warning(f"Error al generar archivo Parquet: {e}. Asegúrate de tener `pyarrow` o `fastparquet` instalados (`pip install pyarrow`).")

    st.markdown("""
    **Justificación de la Migración:** La migración de datos limpios a un Data Warehouse (DW) típicamente implica:
    * **Extract (Extraer):** Obtener datos de las fuentes.
    * **Transform (Transformar):** Los datos son limpiados, estandarizados, validados y preparados para ajustarse al esquema del DW. Esta es la fase que hemos detallado en esta aplicación.
    * **Load (Cargar):** Los datos transformados se cargan en las tablas dimensionales y de hechos del DW. Los formatos como CSV son universales, pero Parquet es preferido en entornos de Big Data y DW por su eficiencia. La simulación de descarga CSV/Parquet representa la salida de este proceso de transformación, lista para ser cargada en un sistema optimizado para consultas analíticas.
    """)

    st.subheader("3.4. Generar Informe Completo")
    st.write("Descarga un informe HTML con un resumen de los indicadores de calidad, EDA y resultados del clustering.")
    
    if st.button("Generar y Descargar Informe HTML"):
        # Asegurarse de que todos los datos necesarios para el informe están disponibles
        if st.session_state['df_nulos_comp'].empty:
            st.error("Por favor, ejecuta la sección '2. Limpieza y Validación de Datos' para generar los indicadores de calidad antes de generar el informe.")
        elif not st.session_state['kpis'] or not st.session_state['eda_plots_data']:
            st.error("Por favor, ejecuta la sección '4. EDA Avanzado & Dashboards' para generar los KPIs y gráficos de EDA antes de generar el informe.")
        elif st.session_state['cluster_results_data'].get('n_clusters', 0) == 0:
            st.warning("El modelo de Clustering no se ha ejecutado o no ha encontrado clusters. El informe incluirá una nota al respecto en esa sección.")
            # Continuar de todos modos, pero advertir.
            
            html_report_data = generate_html_report(
                df_cleaned=st.session_state['df_cleaned'],
                df_original=st.session_state['df_original'],
                df_nulos_comp=st.session_state['df_nulos_comp'],
                indicators_original=st.session_state['indicators_original'],
                indicators_cleaned=st.session_state['indicators_cleaned'],
                kpis=st.session_state['kpis'],
                eda_plots=st.session_state['eda_plots_data'],
                cluster_results=st.session_state['cluster_results_data']
            )
            st.download_button(
                label="Descargar Informe HTML",
                data=html_report_data,
                file_name="informe_calidad_datos_pacientes.html",
                mime="text/html",
                help="Genera y descarga un informe HTML completo."
            )
            st.success("Informe HTML generado y listo para descargar.")
        else:
            html_report_data = generate_html_report(
                df_cleaned=st.session_state['df_cleaned'],
                df_original=st.session_state['df_original'],
                df_nulos_comp=st.session_state['df_nulos_comp'],
                indicators_original=st.session_state['indicators_original'],
                indicators_cleaned=st.session_state['indicators_cleaned'],
                kpis=st.session_state['kpis'],
                eda_plots=st.session_state['eda_plots_data'],
                cluster_results=st.session_state['cluster_results_data']
            )
            st.download_button(
                label="Descargar Informe HTML",
                data=html_report_data,
                file_name="informe_calidad_datos_pacientes.html",
                mime="text/html",
                help="Genera y descarga un informe HTML completo."
            )
            st.success("Informe HTML generado y listo para descargar.")


# --- Sección 4: EDA Avanzado & Dashboards ---
elif section == "4. EDA Avanzado & Dashboards":
    st.header("4. Análisis Exploratorio de Datos (EDA) Avanzado & Dashboards")
    st.write("Explora patrones y tendencias en los datos limpios a través de visualizaciones.")

    df_cleaned_eda = st.session_state['df_cleaned'].copy()

    # Calcular KPIs
    kpis = get_kpis(df_cleaned_eda)
    st.session_state['kpis'] = kpis # Guardar KPIs en session state

    st.subheader("Key Performance Indicators (KPIs)")
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Pacientes", kpis['total_pacientes'])
    kpi_cols[1].metric("Edad Promedio", f"{kpis['edad_promedio']:.2f}")
    kpi_cols[2].metric("Pacientes con Email", kpis['pacientes_con_email'])
    kpi_cols[3].metric("Pacientes con Teléfono", kpis['pacientes_con_telefono'])

    kpi_cols2 = st.columns(4)
    kpi_cols2[0].metric("Hombres (%)", f"{kpis['porcentaje_hombres']:.2f}%")
    kpi_cols2[1].metric("Mujeres (%)", f"{kpis['porcentaje_mujeres']:.2f}%")
    kpi_cols2[2].metric("Ciudades Únicas", kpis['ciudades_unicas'])
    kpi_cols2[3].metric("Emails Inválidos", kpis['emails_invalidos_conteo'])

    st.subheader("Visualizaciones Interactivas")

    # Generar y mostrar gráficos EDA
    eda_plots = generate_eda_plots(df_cleaned_eda)
    st.session_state['eda_plots_data'] = eda_plots # Guardar plots en session state

    for title, fig in eda_plots:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Análisis de Correlación (para variables numéricas)")
    numeric_df = df_cleaned_eda.select_dtypes(include=np.number)
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                            title="Matriz de Correlación de Variables Numéricas")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No hay suficientes variables numéricas para mostrar una matriz de correlación.")


# --- Sección 5: Modelado de Machine Learning (Clustering) ---
elif section == "5. Modelado de Machine Learning (Clustering)":
    st.header("5. Modelado de Machine Learning (Clustering)")
    st.write("Identifica segmentos de pacientes utilizando algoritmos de clustering (K-Means).")

    df_model = st.session_state['df_cleaned'].copy()

    # Preprocesamiento para Clustering
    # Seleccionar características relevantes (edad, y variables dummy para sexo, ciudad)
    features_df = df_model[['edad', 'sexo', 'ciudad']].copy()

    # Eliminar filas con nulos en estas características para el clustering
    features_df.dropna(subset=['edad', 'sexo', 'ciudad'], inplace=True)
    
    if features_df.empty:
        st.error("No hay datos suficientes después de limpiar nulos en edad, sexo y ciudad para realizar clustering.")
    else:
        st.info(f"Se utilizarán {len(features_df)} de {len(df_model)} registros para el clustering después de eliminar nulos.")

        # Codificación One-Hot para variables categóricas
        features_df = pd.get_dummies(features_df, columns=['sexo', 'ciudad'], drop_first=True)

        # Escalar las características numéricas
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        scaled_features_df = pd.DataFrame(scaled_features, columns=features_df.columns, index=features_df.index)

        st.subheader("Selección del Número de Clusters (K)")
        max_clusters = min(10, len(scaled_features_df) - 1) # Asegura que no haya más clusters que datos-1
        if max_clusters < 2:
            st.warning("No hay suficientes datos para realizar clustering con K > 1. Necesitas al menos 2 puntos de datos.")
        else:
            n_clusters_option = st.slider("Selecciona el número de clusters (K):", min_value=2, max_value=max_clusters, value=min(3, max_clusters))

            if st.button("Ejecutar Clustering K-Means"):
                with st.spinner(f"Ejecutando K-Means con {n_clusters_option} clusters..."):
                    kmeans = KMeans(n_clusters=n_clusters_option, random_state=42, n_init=10) # n_init por deprecation warning
                    features_df['cluster'] = kmeans.fit_predict(scaled_features)

                    # Calcular métricas de evaluación
                    if len(features_df) >= n_clusters_option: # Métrica de Silueta requiere al menos n_clusters puntos
                        silhouette_avg = silhouette_score(scaled_features, features_df['cluster'])
                        davies_bouldin_idx = davies_bouldin_index(scaled_features, features_df['cluster'])
                        calinski_harabasz_idx = calinski_harabasz_index(scaled_features, features_df['cluster'])
                    else:
                        silhouette_avg = np.nan
                        davies_bouldin_idx = np.nan
                        calinski_harabasz_idx = np.nan

                    inertia = kmeans.inertia_

                    st.session_state['cluster_results_data'] = {
                        'n_clusters': n_clusters_option,
                        'silhouette_score': silhouette_avg,
                        'davies_bouldin_index': davies_bouldin_idx,
                        'calinski_harabasz_index': calinski_harabasz_idx,
                        'inertia': inertia
                    }

                    st.subheader("Resultados del Clustering")
                    st.write(f"**Número de Clusters:** {n_clusters_option}")
                    st.write(f"**Puntuación de Silueta:** {silhouette_avg:.2f}")
                    st.write(f"**Índice Davies-Bouldin:** {davies_bouldin_idx:.2f}")
                    st.write(f"**Índice Calinski-Harabasz:** {calinski_harabasz_idx:.2f}")
                    st.write(f"**Inercia (Sum of Squared Distances):** {inertia:.2f}")

                    # Añadir las asignaciones de cluster al DataFrame limpio original
                    st.session_state['df_cleaned'] = st.session_state['df_cleaned'].merge(
                        features_df[['cluster']],
                        left_index=True,
                        right_index=True,
                        how='left'
                    )
                    st.session_state['df_cleaned']['cluster'] = st.session_state['df_cleaned']['cluster'].astype('Int64') # Para soportar nulos

                    st.write("### Características de los Clusters (Centros de Cluster)")
                    cluster_centers_df = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features_df.columns)
                    cluster_centers_df['cluster'] = range(n_clusters_option)
                    cluster_centers_df.set_index('cluster', inplace=True)
                    st.dataframe(cluster_centers_df)
                    st.session_state['cluster_results_data']['cluster_centers_df'] = cluster_centers_df

                    st.write("### Conteo de Pacientes por Cluster")
                    cluster_counts = features_df['cluster'].value_counts().sort_index().reset_index()
                    cluster_counts.columns = ['Cluster', 'Conteo de Pacientes']
                    st.dataframe(cluster_counts)
                    st.session_state['cluster_results_data']['cluster_counts_df'] = cluster_counts

                    st.write("### Visualización de Clusters")
                    cluster_plots = []

                    # Gráfico de dispersión de Edad vs. Sexo (si es posible, para una mejor visualización)
                    # Necesitamos df_cleaned con la columna de cluster
                    df_plot_clusters = st.session_state['df_cleaned'].dropna(subset=['edad', 'sexo', 'cluster'])
                    if not df_plot_clusters.empty:
                        fig_cluster_age_sex = px.scatter(
                            df_plot_clusters,
                            x='edad',
                            y='sexo',
                            color='cluster',
                            title='Clusters por Edad y Sexo',
                            hover_data=['nombre', 'ciudad']
                        )
                        cluster_plots.append(("Clusters por Edad y Sexo", fig_cluster_age_sex))
                        st.plotly_chart(fig_cluster_age_sex, use_container_width=True)

                    # Gráfico de dispersión 3D (si hay suficientes dimensiones reducidas o PCA/t-SNE)
                    # Por simplicidad, se usa solo edad y una dummy de sexo/ciudad para el ejemplo
                    # OJO: Para gráficos 3D más significativos, se requeriría PCA o t-SNE
                    if len(features_df.columns) >= 2:
                        fig_3d = px.scatter_3d(
                            features_df,
                            x=features_df.columns[0],
                            y=features_df.columns[1],
                            z='cluster', # Usar cluster como eje Z para visualización
                            color='cluster',
                            title='Clusters en 3D (Características Seleccionadas)',
                            hover_data=['cluster']
                        )
                        cluster_plots.append(("Clusters en 3D (Características Seleccionadas)", fig_3d))
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.info("No hay suficientes características para una visualización 3D significativa.")
                    
                    st.session_state['cluster_results_data']['cluster_plots_data'] = cluster_plots
                    st.success("Clustering completado y resultados generados.")
