import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
import numpy as np # Importar numpy para np.nan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Initial Configuration ---
st.set_page_config(layout="wide", page_title="Análisis de Calidad de Datos Hospital")
st.title("🏥 Análisis y Calidad de Datos de Pacientes de Hospital")
st.markdown("Esta aplicación realiza un análisis exhaustivo de la calidad de los datos de pacientes, seguido de procesos de limpieza, validación, generación de KPIs, EDA avanzado y un modelo de Machine Learning.")

# URL del archivo JSON
DATA_URL_PACIENTES = "https://raw.githubusercontent.com/JulianTorrest/hospital/refs/heads/main/dataset_hospital%202.json"

# --- Funciones de Ayuda para Carga de Datos y Caching ---
@st.cache_data
def load_data(url, key_name):
    """Carga datos desde una URL y normaliza un JSON si tiene una clave raíz."""
    try:
        response = requests.get(url)
        response.raise_for_status() # Lanza un error para códigos de estado HTTP malos
        data = response.json()
        if key_name in data:
            df = pd.json_normalize(data[key_name])
        else:
            df = pd.DataFrame(data) # Asume que es una lista de objetos directamente
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error de red al cargar datos desde {url}: {e}")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"Error al decodificar JSON desde {url}: {e}")
        return pd.DataFrame()
    except KeyError:
        st.error(f"La clave '{key_name}' no se encontró en el JSON de {url}. Asegúrate de que el JSON tenga una estructura con '{key_name}'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al procesar datos desde {url}: {e}")
        return pd.DataFrame()

# Función para calcular la edad (usada en limpieza y validación)
def calculate_age_from_dob(row_dob, current_date):
    if pd.isna(row_dob):
        return None
    else:
        # Asegurarse de que row_dob sea un objeto datetime, no solo una cadena
        if isinstance(row_dob, str):
            try:
                row_dob = datetime.strptime(row_dob, '%Y-%m-%d').date()
            except ValueError:
                return None # No se pudo analizar la fecha
        elif isinstance(row_dob, pd.Timestamp):
            row_dob = row_dob.date()

        # Calcular la edad
        age = current_date.year - row_dob.year - ((current_date.month, current_date.day) < (row_dob.month, row_dob.day))
        return age if age >= 0 else None # La edad no puede ser negativa

# Cargar datos de pacientes
df_pacientes = load_data(DATA_URL_PACIENTES, 'pacientes')

# --- Sidebar para Navegación ---
st.sidebar.header("Navegación")
selected_section = st.sidebar.radio(
    "Ir a la sección:",
    ("1. Exploración Inicial", "2. Limpieza y Validación", "3. Indicadores y Documentación", "4. EDA Avanzado & Dashboards", "5. Modelado de Machine Learning")
)

# --- Contenido Principal de la Aplicación ---

# Sección 1: Análisis de Calidad de Datos (Exploración)
if selected_section == "1. Exploración Inicial":
    st.header("1. 📉 Análisis de Calidad de Datos (Exploración)")
    st.markdown("Identificación de los principales problemas de calidad en la tabla de pacientes.")

    if df_pacientes.empty:
        st.warning("No se pudieron cargar los datos de pacientes o el DataFrame está vacío.")
    else:
        st.subheader("1.1. Vista Previa de Datos Originales")
        st.dataframe(df_pacientes.head())

        st.subheader("1.2. Información General y Tipos de Datos")
        buffer = pd.io.common.StringIO()
        df_pacientes.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("1.3. Valores Faltantes (Nulos)")
        missing_values = df_pacientes.isnull().sum()
        missing_percentage = (df_pacientes.isnull().sum() / len(df_pacientes)) * 100
        missing_df = pd.DataFrame({
            'Valores Faltantes': missing_values,
            'Porcentaje (%)': missing_percentage
        }).sort_values(by='Porcentaje (%)', ascending=False)
        st.dataframe(missing_df[missing_df['Valores Faltantes'] > 0])

        if not missing_df[missing_df['Valores Faltantes'] > 0].empty:
            st.markdown("""
            **Observaciones Iniciales sobre Valores Faltantes:**
            - **`edad`**: Muestra `null` en el JSON para algunos registros. Esto es un problema, ya que la edad es crucial y puede calcularse a partir de la fecha de nacimiento.
            - **`fecha_nacimiento`**: Aunque no hay nulos directos, es importante verificar el formato y la validez de la fecha.
            """)
        else:
            st.info("No se detectaron valores faltantes significativos en los datos cargados.")

        st.subheader("1.4. Inconsistencias y Formatos")

        st.markdown("#### Columna `sexo`")
        st.write(df_pacientes['sexo'].value_counts(dropna=False))
        # Convertir a minúsculas para verificar inconsistencias más fácilmente
        sex_lower_unique = df_pacientes['sexo'].astype(str).str.lower().unique()
        # Verificar si hay más de dos categorías únicas (sin contar NaN) o si hay 'f'/'m' que necesitan mapeo
        if len(sex_lower_unique[sex_lower_unique != 'nan']) > 2 or any(s in ['f', 'm'] for s in sex_lower_unique):
            st.warning("Problema: Inconsistencia en la capitalización o variaciones en la columna `sexo` (ej., 'Female' vs 'female', 'F' vs 'f', 'M' vs 'm', u otros valores inesperados).")
        else:
            st.info("La columna `sexo` parece estar relativamente estandarizada o con pocas inconsistencias (requerirá limpieza).")


        st.markdown("#### Columna `fecha_nacimiento`")
        # Verificar formatos de fecha inválidos
        invalid_dates = df_pacientes[pd.to_datetime(df_pacientes['fecha_nacimiento'], errors='coerce').isna() & df_pacientes['fecha_nacimiento'].notna()]
        if not invalid_dates.empty:
            st.warning(f"Problema: Se encontraron **{len(invalid_dates)}** fechas de nacimiento con formato inválido.")
            st.dataframe(invalid_dates)
        else:
            st.info("No se encontraron formatos de fecha de nacimiento inválidos aparentes.")

        st.markdown("#### Columna `email`")
        # Validación básica de email
        invalid_emails = df_pacientes[~df_pacientes['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        if not invalid_emails.empty:
            st.warning(f"Problema: Se encontraron **{len(invalid_emails)}** correos electrónicos con formato potencialmente inválido.")
            st.dataframe(invalid_emails.head())
        else:
            st.info("No se encontraron formatos de correo electrónico inválidos aparentes (validación básica).")

        st.markdown("#### Columna `telefono`")
        # Validación básica de teléfono (solo si no es nulo y no es un número después de limpiar no-dígitos)
        # Primero, intentar limpiar para verificar si persisten los problemas.
        temp_telefono_cleaned = df_pacientes['telefono'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        non_numeric_phones_after_temp_clean = df_pacientes[temp_telefono_cleaned.notna() & ~temp_telefono_cleaned.str.isdigit()]
        if not non_numeric_phones_after_temp_clean.empty:
            st.warning(f"Problema: Se encontraron **{len(non_numeric_phones_after_temp_clean)}** números de teléfono con caracteres no numéricos o que no se convierten a un formato numérico válido.")
            st.dataframe(non_numeric_phones_after_temp_clean.head())
        else:
            st.info("No se encontraron números de teléfono no numéricos aparentes.")


        st.markdown("""
        ### **Resumen de Problemas de Calidad (Pacientes):**

        1.  **Valores Nulos:** Principalmente en la columna `edad`.
        2.  **Inconsistencias de Formato:**
            * `sexo`: Posibles variaciones en la capitalización (`Female` vs `female`), o abreviaciones (`F` vs `M`).
            * `fecha_nacimiento`: Necesita conversión a tipo `datetime` y manejo de posibles formatos incorrectos.
            * `edad`: Debe ser un valor numérico y consistente con `fecha_nacimiento`. Si es `null`, debe ser calculado.
            * `email`, `telefono`: Requieren validación de formato (aunque el ejemplo dado parece limpio, es buena práctica).
        3.  **Coherencia de Datos:** `edad` debe ser derivable de `fecha_nacimiento` y ser un número positivo.
        """)
        st.info("Nota: Dado que solo tenemos la tabla 'pacientes' de la URL, el análisis se centra en ella.")


# Sección 2: Limpieza y Validación
elif selected_section == "2. Limpieza y Validación":
    st.header("2. 🧹 Limpieza y Validación")
    st.markdown("Aplicación de un proceso de limpieza para resolver los problemas identificados y validaciones cruzadas.")

    if df_pacientes.empty:
        st.warning("No se pudieron cargar los datos de pacientes para la limpieza.")
    else:
        df_cleaned = df_pacientes.copy() # Trabajar en una copia para evitar modificar el original
        current_date = date.today() # Definir la fecha actual una vez

        # --- Limpieza de Datos ---
        st.subheader("2.1. Proceso de Limpieza")

        st.markdown("#### Limpieza de `sexo`")
        st.code("""
# Convertir a cadena y a minúsculas para un manejo consistente
df_cleaned['sexo'] = df_cleaned['sexo'].astype(str).str.lower()

# Mapear valores a 'Female', 'Male' o np.nan para no mapeados
sex_mapping = {
    'f': 'Female',
    'female': 'Female',
    'm': 'Male',
    'male': 'Male'
}
df_cleaned['sexo'] = df_cleaned['sexo'].map(sex_mapping) # Esto generará np.nan para valores no mapeados

# Finalmente, reemplazar np.nan con None (Python None)
df_cleaned.loc[df_cleaned['sexo'].isna(), 'sexo'] = None
""")
        # Aplicar la lógica de limpieza y mapeo
        df_cleaned['sexo'] = df_cleaned['sexo'].astype(str).str.lower()
        sex_mapping = {
            'f': 'Female',
            'female': 'Female',
            'm': 'Male',
            'male': 'Male'
        }
        df_cleaned['sexo'] = df_cleaned['sexo'].map(sex_mapping) # Genera np.nan para no mapeados
        
        # Rellenar np.nan (si los hay) con Python None
        df_cleaned.loc[df_cleaned['sexo'].isna(), 'sexo'] = None

        st.write("Valores de `sexo` después de la normalización y mapeo:")
        st.write(df_cleaned['sexo'].value_counts(dropna=False))
        st.markdown("""
        **Justificación:** Los valores de la columna `sexo` se normalizan a minúsculas y luego se mapean explícitamente a `'Female'` o `'Male'`. Cualquier valor que no coincida con estas categorías mapeadas (incluyendo cadenas vacías, 'nan', o 'Other') se convierte a `None` (nulo), asegurando una consistencia total para análisis y filtros. Se utiliza `numpy.nan` para el manejo intermedio de nulos, que es la forma estándar de Pandas.
        """)

        st.markdown("#### Limpieza y Cálculo de `fecha_nacimiento` y `edad`")
        st.code("""
# Convertir 'fecha_nacimiento' a datetime, forzando nulos si el formato es inválido
df_cleaned['fecha_nacimiento'] = pd.to_datetime(df_cleaned['fecha_nacimiento'], errors='coerce')

# Calcular 'edad' para nulos o valores inconsistentes
current_date = date.today() # Fecha actual
df_cleaned['edad_calculada'] = df_cleaned['fecha_nacimiento'].apply(lambda dob: calculate_age_from_dob(dob, current_date))

# Priorizar edad calculada si fecha_nacimiento es válida, de lo contrario usar existente o None
df_cleaned['edad'] = df_cleaned.apply(
    lambda row: row['edad_calculada'] if pd.notna(row['edad_calculada']) else row['edad'], axis=1
)
df_cleaned['edad'] = df_cleaned['edad'].astype('Int64') # Int64 para permitir NaNs y mantenerlo entero
""")
        # Aplicar limpieza de fecha y edad
        df_cleaned['fecha_nacimiento'] = pd.to_datetime(df_cleaned['fecha_nacimiento'], errors='coerce')
        df_cleaned['edad_calculada'] = df_cleaned['fecha_nacimiento'].apply(lambda dob: calculate_age_from_dob(dob, current_date))
        df_cleaned['edad'] = df_cleaned.apply(
            lambda row: row['edad_calculada'] if pd.notna(row['edad_calculada']) else row['edad'], axis=1
        )
        df_cleaned['edad'] = df_cleaned['edad'].astype('Int64')
        df_cleaned = df_cleaned.drop(columns=['edad_calculada']) # Eliminar columna temporal
        st.write("Valores nulos en `edad` después de la limpieza:", df_cleaned['edad'].isna().sum())
        st.write("Valores nulos en `fecha_nacimiento` después de la limpieza:", df_cleaned['fecha_nacimiento'].isna().sum())
        st.markdown("""
        **Justificación:**
        - `fecha_nacimiento` se convierte a tipo `datetime`, convirtiendo formatos inválidos a `NaT` (Not a Time).
        - `edad` se recalcula basándose en `fecha_nacimiento` si es válida. Esta edad calculada se prioriza si está disponible. Si `fecha_nacimiento` es `NaT`, se mantiene la `edad` original.
        - Asegura que la edad sea un entero no negativo. Se usa `Int64` para manejar nulos en columnas numéricas.
        """)

        st.markdown("#### Limpieza de `telefono`")
        st.code("""
# Eliminar caracteres no numéricos
df_cleaned['telefono'] = df_cleaned['telefono'].astype(str).str.replace(r'[^0-9]', '', regex=True)
# Reemplazar cadenas vacías (o solo espacios) con None
df_cleaned.loc[df_cleaned['telefono'].str.strip() == '', 'telefono'] = None
""")
        df_cleaned['telefono'] = df_cleaned['telefono'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df_cleaned.loc[df_cleaned['telefono'].str.strip() == '', 'telefono'] = None # Reemplazar cadenas vacías con None
        st.write("Ejemplos de `telefono` después de la limpieza:")
        st.dataframe(df_cleaned['telefono'].head())
        st.markdown("**Justificación:** Se eliminan caracteres no numéricos del teléfono para estandarizar el formato. Las cadenas vacías o aquellas con solo espacios se convierten a `None`.")

        st.subheader("2.2. Validaciones Cruzadas")
        st.markdown("Se aplican reglas para asegurar la consistencia lógica entre columnas.")

        st.markdown("#### Validación: `edad` consistente con `fecha_nacimiento`")
        # Recalcular edad para comparar con la edad final limpia
        df_cleaned_temp_age_check = df_cleaned.copy()
        df_cleaned_temp_age_check['calculated_age_for_check'] = df_cleaned_temp_age_check['fecha_nacimiento'].apply(lambda dob: calculate_age_from_dob(dob, current_date))

        inconsistent_ages = df_cleaned[
            (df_cleaned['edad'].notna()) &
            (df_cleaned_temp_age_check['calculated_age_for_check'].notna()) &
            (abs(df_cleaned['edad'] - df_cleaned_temp_age_check['calculated_age_for_check']) > 1) # Tolerancia de 1 año por posibles discrepancias de actualización
        ]
        if not inconsistent_ages.empty:
            st.warning(f"Se encontraron **{len(inconsistent_ages)}** registros con **edad inconsistente** con la fecha de nacimiento (diferencia > 1 año) *después de la limpieza*.")
            st.dataframe(inconsistent_ages[['id_paciente', 'fecha_nacimiento', 'edad']].head())
            st.markdown("""
            **Regla de Validación:** La edad calculada a partir de `fecha_nacimiento` debe ser consistente con la `edad` reportada (se permite una pequeña tolerancia para posibles discrepancias de actualización de fechas).
            **Acción:** La limpieza ya prioriza la edad calculada si `fecha_nacimiento` es válida, minimizando estas inconsistencias. Si aún existen, podría indicar un `fecha_nacimiento` erróneo.
            """)
        else:
            st.success("No se encontraron inconsistencias significativas entre `edad` y `fecha_nacimiento` después de la limpieza.")

        st.markdown("#### Validación: `email` con formato válido")
        invalid_email_after_clean = df_cleaned[~df_cleaned['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        if not invalid_email_after_clean.empty:
            st.warning(f"Se encontraron **{len(invalid_email_after_clean)}** registros con **correo electrónico inválido** después de la limpieza.")
            st.dataframe(invalid_email_after_clean[['id_paciente', 'email']].head())
            st.markdown("""
            **Regla de Validación:** El campo `email` debe seguir un formato de correo electrónico estándar (`texto@texto.dominio`).
            **Acción:** Se identifican pero no se modifican automáticamente, ya que esto requeriría inferencia o interacción manual.
            """)
        else:
            st.success("Todos los correos electrónicos parecen tener un formato válido después de la limpieza (validación básica).")

        st.markdown("#### Validación: `telefono` contiene solo dígitos (después de la limpieza)")
        non_numeric_phones_cleaned = df_cleaned[df_cleaned['telefono'].notna() & ~df_cleaned['telefono'].astype(str).str.isdigit()]
        if not non_numeric_phones_cleaned.empty:
            st.warning(f"Se encontraron **{len(non_numeric_phones_cleaned)}** registros con **números de teléfono que contienen caracteres no numéricos** después de la limpieza (esto no debería ocurrir si la limpieza fue efectiva).")
            st.dataframe(non_numeric_phones_cleaned[['id_paciente', 'telefono']].head())
        else:
            st.success("Todos los teléfonos contienen solo dígitos o son nulos después de la limpieza.")

        st.subheader("2.3. DataFrame Después de la Limpieza")
        st.write("Las primeras 10 filas del DataFrame limpio:")
        st.dataframe(df_cleaned.head(10))
        st.write("Información del DataFrame limpio:")
        buffer_cleaned = pd.io.common.StringIO()
        df_cleaned.info(buf=buffer_cleaned)
        s_cleaned = buffer_cleaned.getvalue()
        st.text(s_cleaned)

        # Guardar el DataFrame limpio y el original en el estado de la sesión para usar en la siguiente sección
        st.session_state['df_cleaned'] = df_cleaned
        st.session_state['df_original'] = df_pacientes

# Sección 3: Indicadores de Calidad y Documentación
elif selected_section == "3. Indicadores y Documentación":
    st.header("3. 📈 Indicadores de Calidad y Documentación")
    st.markdown("Resumen de indicadores de calidad antes y después de la limpieza, junto con la documentación.")

    if 'df_cleaned' not in st.session_state or 'df_original' not in st.session_state:
        st.warning("Por favor, navega primero a la sección 'Limpieza y Validación' para generar los datos limpios y el estado de la sesión.")
    else:
        df_original = st.session_state['df_original']
        df_cleaned = st.session_state['df_cleaned']

        st.subheader("3.1. Indicadores de Calidad de Datos")

        # Función para calcular indicadores de calidad
        def get_quality_indicators(df):
            total_rows = len(df)
            missing_values = df.isnull().sum()
            missing_percentage = (missing_values / total_rows) * 100
            data_types = df.dtypes

            indicators = {
                'Total Registros': total_rows,
                'Valores Nulos por Columna (%)': missing_percentage.to_dict(),
                'Tipos de Datos por Columna': {col: str(dtype) for col, dtype in data_types.items()}
            }
            return indicators

        indicators_original = get_quality_indicators(df_original)
        indicators_cleaned = get_quality_indicators(df_cleaned)

        st.markdown("#### Comparación de Valores Faltantes (%)")
        cols = ['edad', 'fecha_nacimiento', 'telefono', 'sexo'] # Columnas relevantes para nulos
        data_nulos = {
            'Columna': cols,
            'Original (%)': [indicators_original['Valores Nulos por Columna (%)'].get(col, 0) for col in cols],
            'Limpio (%)': [indicators_cleaned['Valores Nulos por Columna (%)'].get(col, 0) for col in cols]
        }
        df_nulos_comp = pd.DataFrame(data_nulos)
        st.dataframe(df_nulos_comp.set_index('Columna'))

        st.markdown("""
        **Observaciones:**
        - Se espera una **reducción significativa** en el porcentaje de nulos en `edad` si `fecha_nacimiento` estaba disponible y era válida.
        - `fecha_nacimiento` puede mostrar un aumento de nulos si los formatos originales eran inválidos y se convirtieron a `NaT`.
        - `telefono` puede tener nulos si quedaron cadenas vacías después de limpiar caracteres no numéricos.
        - `sexo` podría tener nulos si había valores vacíos o no estandarizables.
        """)

        st.markdown("#### Comparación de Tipos de Datos")
        st.write("Tipos de datos originales:")
        st.json(indicators_original['Tipos de Datos por Columna'])
        st.write("Tipos de datos después de la limpieza:")
        st.json(indicators_cleaned['Tipos de Datos por Columna'])
        st.markdown("""
        **Observaciones:**
        - `fecha_nacimiento` debería cambiar de `object` (cadena) a `datetime64[ns]` (tipo fecha y hora).
        - `edad` debería cambiar de `object` (si contenía nulos o estaba mezclado) o `float64` (si se infirió numérico) a `Int64` (entero con soporte para nulos).
        - `telefono` y `email` idealmente deberían permanecer como `object` (cadena) pero con formato validado.
        """)

        st.markdown("#### Indicadores de Consistencia y Unicidad")
        st.write("**`sexo` - Unicidad de Categorías:**")
        st.write(f"Original: {df_original['sexo'].value_counts(dropna=False).index.tolist()}")
        st.write(f"Limpio: {df_cleaned['sexo'].value_counts(dropna=False).index.tolist()}")
        st.markdown("""
        **Observación:** Se espera que el número de categorías únicas y sus nombres se normalicen después de la limpieza (ej., solo 'Female', 'Male' y `None`).
        """)

        st.write("**`email` - Patrón de Formato (Conteo de Inválidos):**")
        invalid_emails_original = df_original[~df_original['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        invalid_emails_cleaned = df_cleaned[~df_cleaned['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        st.write(f"Correos inválidos (Original): **{len(invalid_emails_original)}**")
        st.write(f"Correos inválidos (Limpio): **{len(invalid_emails_cleaned)}**")
        st.markdown("""
        **Observación:** Aunque la limpieza no los altera, se validó su formato. Este indicador muestra si persisten correos con formato no estándar.
        """)

        st.write("**`telefono` - Contiene solo dígitos (después de la limpieza):**")
        non_numeric_phones_cleaned = df_cleaned[df_cleaned['telefono'].notna() & ~df_cleaned['telefono'].astype(str).str.isdigit()]
        if not non_numeric_phones_cleaned.empty:
            st.warning(f"Se encontraron **{len(non_numeric_phones_cleaned)}** registros con **números de teléfono que contienen caracteres no numéricos** después de la limpieza (esto no debería ocurrir si la limpieza fue efectiva).")
        else:
            st.success("Todos los teléfonos contienen solo dígitos o son nulos después de la limpieza.")

        st.subheader("3.2. Documentación del Proceso")

        st.markdown("### **Supuestos Adoptados Durante la Limpieza:**")
        st.markdown("""
        * **Fuente Única para Edad:** Se asume que `fecha_nacimiento` es la fuente más confiable para determinar la `edad`. Si `fecha_nacimiento` es válida, se **prioriza el cálculo de la edad a partir de ella** sobre el valor `edad` existente si este es nulo o inconsistente. La edad se calcula como la diferencia en años a la fecha actual, ajustando por mes y día.
        * **Formato de `sexo`:** Se asume que los valores `Female`, `female`, `Male`, `male`, `F`, `f`, `M`, `m` y sus variaciones deben ser estandarizados a **`Female`** y **`Male`**. Cualquier otro valor (`NaN`, vacío, o no reconocido) se convierte a `None`.
        * **Formato de `telefono`:** Se asume que los números de teléfono deben contener solo dígitos. Cualquier otro carácter (guiones, espacios, paréntesis, etc.) es **removido**. Las cadenas vacías o que solo consisten en espacios resultantes de esta limpieza se interpretan como nulas (`None`).
        * **Coherencia de Fechas:** Se asume que las fechas de nacimiento no pueden ser en el futuro ni excesivamente antiguas (la edad se calcula en relación con la fecha actual y las edades negativas se descartan, convirtiéndolas a `None`).
        * **ID de Paciente:** Se asume que `id_paciente` es el identificador **único** de cada paciente y no se espera que tenga problemas de calidad (duplicados, nulos).
        """)

        st.markdown("### **Reglas de Validación Implementadas:**")
        st.markdown("""
        * **Validación de `fecha_nacimiento`:** Se verifica que la columna pueda ser convertida a tipo `datetime`. Los valores que no cumplen con este formato se marcan como `NaT` (Not a Time).
        * **Validación de `edad`:**
            * Debe ser un entero no negativo.
            * Debe ser **consistente** con `fecha_nacimiento`: la `edad` calculada a partir de `fecha_nacimiento` debe ser cercana a la `edad` reportada (se permite una tolerancia de 1 año para posibles discrepancias de actualización de fechas en los datos originales).
        * **Validación de `sexo`:** Los valores deben estar dentro de un conjunto predefinido de categorías estandarizadas (`Female`, `Male` o `None`).
        * **Validación de `email`:** Se verifica que el formato siga una expresión regular básica (`[^@]+@[^@]+\.[^@]+`) para asegurar que contenga un `@` y al menos un `.` en el dominio. Esta es una validación de patrón, no de existencia.
        * **Validación de `telefono`:** Se verifica que, después de la limpieza, la columna contenga solo caracteres numéricos (o sea nula).
        """)

        st.markdown("### **Recomendaciones de Mejora para Asegurar la Calidad Futura de los Datos:**")
        st.markdown("""
        1.  **Validación en Origen:** Implementar validaciones a nivel de entrada de datos (ej., formularios web, bases de datos) para `fecha_nacimiento`, `sexo`, `email` y `telefono`.
            * **`fecha_nacimiento`:** Usar selectores de fecha para prevenir entradas manuales erróneas y asegurar formato `AAAA-MM-DD`.
            * **`sexo`:** Usar listas desplegables con opciones predefinidas (`Female`, `Male`) para evitar inconsistencias de capitalización o errores tipográficos.
            * **`email`:** Implementar validación de formato de correo electrónico en tiempo real en la entrada de datos y, si es posible, una verificación de dominio.
            * **`telefono`:** Forzar la entrada de solo dígitos o un formato específico (ej., con máscaras de entrada) dependiendo del país, y validar longitud mínima/máxima.
        2.  **Estandarización de `ciudad`:** Implementar un catálogo o lista maestra de ciudades/municipios para asegurar consistencia y evitar variaciones en los nombres de ciudades (ej., "Barranquilla" vs "barranquilla", o errores tipográficos).
        3.  **Definición de Campos Obligatorios:** Establecer claramente qué campos son obligatorios (ej., `id_paciente`, `nombre`, `fecha_nacimiento`, `sexo`) en la base de datos o sistema de entrada para reducir la aparición de valores nulos críticos.
        4.  **Auditorías Regulares de Datos:** Realizar auditorías periódicas de la base de datos para identificar nuevos patrones de error o degradación de la calidad de los datos con el tiempo.
        5.  **Documentación de Metadatos:** Mantener un `diccionario de datos` actualizado que defina claramente cada campo, su tipo de dato esperado, formato, reglas de validación y significado, accesible para todo el equipo.
        6.  **Sistema de Reporte de Errores:** Establecer un mecanismo para que los usuarios (personal del hospital, médicos) reporten inconsistencias o errores en los datos cuando los detecten, con un flujo claro para su corrección.
        7.  **Capacitación del Personal:** Asegurar que el personal encargado de la entrada de datos esté continuamente capacitado en las mejores prácticas de entrada de datos y comprenda la importancia de la calidad de los datos para la toma de decisiones y la atención al paciente.
        """)

        st.subheader("3.3. Bonus (Opcional)")
        st.markdown("""
        #### Implementación de Pruebas Automáticas
        Para implementar pruebas automáticas para la calidad de los datos, se podrían usar frameworks como **Pytest** o **Great Expectations**.

        **Ejemplo conceptual con Pytest (en un archivo `tests/test_data_quality.py`):**
        ```python
        # Este código es conceptual y no forma parte de app.py
        # Deberías tener tus funciones de limpieza y validación en un módulo separado para importarlas aquí.
        import pandas as pd
        import pytest
        from datetime import date
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
        1.  `pytest` instalado: `pip install pytest`
        2.  Guardar el código de prueba en un archivo como `tests/test_data_quality.py` (o similar) en una carpeta `tests/`.
        3.  **Importante:** Refactorizar tus funciones de limpieza y validación de `app.py` en un módulo de Python separado (ej., `procesamiento_datos.py`) para que puedas importarlas en las pruebas. O, para esta demostración, puedes copiar y adaptar las funciones de limpieza dentro del propio archivo de prueba como se muestra arriba.
        4.  Ejecutar `pytest` en tu terminal desde la raíz de tu proyecto.

        #### Simulación de Migración de Datos Limpios a una Estructura Objetivo
        Una vez que los datos han sido limpiados y validados, el siguiente paso lógico en una tubería de datos es cargarlos en una estructura objetivo, como un Data Warehouse o una base de datos analítica. Formatos como **Parquet** son ideales para esto debido a su naturaleza columnar, compresión eficiente y capacidad para manejar esquemas complejos.

        Aquí simulamos la descarga de los datos limpios en formatos comunes para la migración.
        """)

        # Código para el bonus de descarga
        if 'df_cleaned' in st.session_state:
            st.markdown("#### Simulación de Migración de Datos Limpios")

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            @st.cache_data
            def convert_df_to_parquet(df):
                try:
                    import pyarrow # Asegúrate de que pyarrow esté instalado para to_parquet
                    return df.to_parquet(index=False)
                except ImportError:
                    st.error("Para descargar en formato Parquet, instala 'pyarrow': `pip install pyarrow`")
                    return None
                except Exception as e:
                    st.error(f"Error al generar el archivo Parquet: {e}")
                    return None

            col_csv, col_parquet = st.columns(2)
            with col_csv:
                st.download_button(
                    label="Descargar Datos Limpios (CSV)",
                    data=convert_df_to_csv(st.session_state['df_cleaned']),
                    file_name="pacientes_limpios.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            with col_parquet:
                parquet_data = convert_df_to_parquet(st.session_state['df_cleaned'])
                if parquet_data is not None: # Solo mostrar el botón si los datos parquet se generaron con éxito
                    st.download_button(
                        label="Descargar Datos Limpios (Parquet)",
                        data=parquet_data,
                        file_name="pacientes_limpios.parquet",
                        mime="application/octet-stream",
                        key="download_parquet"
                    )

            st.markdown("""
            **Justificación de la Migración:**
            La migración de datos limpios a un Data Warehouse (DW) típicamente implica:
            1.  **Extract (Extraer):** Obtener datos de las fuentes.
            2.  **Transform (Transformar):** Los datos son limpiados, estandarizados, validados y preparados para ajustarse al esquema del DW. Esta es la fase que hemos detallado en esta aplicación.
            3.  **Load (Cargar):** Los datos transformados se cargan en las tablas dimensionales y de hechos del DW.
            Los formatos como CSV son universales, pero Parquet es preferido en entornos de Big Data y DW por su eficiencia. La simulación de descarga CSV/Parquet representa la salida de este proceso de transformación, lista para ser cargada en un sistema optimizado para consultas analíticas.
            """)

# --- Nueva Sección 4: EDA Avanzado y Dashboards ---
elif selected_section == "4. EDA Avanzado & Dashboards":
    st.header("4. 📊 EDA Avanzado y Dashboards Interactivos")
    st.markdown("Exploración profunda de los datos limpios y creación de visualizaciones interactivas.")

    if 'df_cleaned' not in st.session_state:
        st.warning("Por favor, navega primero a la sección 'Limpieza y Validación' para cargar los datos limpios.")
    else:
        df_display = st.session_state['df_cleaned'].copy()

        st.subheader("Filtros del Dashboard")
        col1, col2, col3 = st.columns(3)

        # Filtro por ciudad
        all_cities = ['Todas'] + sorted(df_display['ciudad'].dropna().unique().tolist())
        selected_city_filter = col1.selectbox("Filtrar por Ciudad:", all_cities)
        if selected_city_filter != 'Todas':
            df_display = df_display[df_display['ciudad'] == selected_city_filter]

        # Filtro por sexo
        # Asegurarse de que 'None' aparezca como una opción si hay nulos
        unique_sexes = df_display['sexo'].dropna().unique().tolist()
        if df_display['sexo'].isnull().any():
            unique_sexes.append('No especificado') # Añadir una opción para nulos
        all_sex = ['Todos'] + sorted(unique_sexes)
        selected_sex_filter = col2.selectbox("Filtrar por Sexo:", all_sex)
        if selected_sex_filter == 'No especificado':
            df_display = df_display[df_display['sexo'].isnull()]
        elif selected_sex_filter != 'Todos':
            df_display = df_display[df_display['sexo'] == selected_sex_filter]

        # Filtro por rango de edad
        # Asegurarse de que el df_display filtrado tenga valores para min/max
        if not df_display['edad'].dropna().empty:
            min_age_data = int(df_display['edad'].min())
            max_age_data = int(df_display['edad'].max())
            age_range = col3.slider("Rango de Edad:", min_value=min_age_data, max_value=max_age_data, value=(min_age_data, max_age_data))
            df_display = df_display[(df_display['edad'] >= age_range[0]) & (df_display['edad'] <= age_range[1])]
        else:
            col3.info("No hay edades disponibles para filtrar.")


        st.subheader("Métricas Clave (KPIs)")
        num_patients = len(df_display)
        avg_age = df_display['edad'].mean()
        most_common_city = df_display['ciudad'].mode()[0] if not df_display['ciudad'].empty else "N/A"

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Total de Pacientes (Filtrados)", num_patients)
        with kpi2:
            st.metric("Edad Promedio (Filtrada)", f"{avg_age:.1f}" if not pd.isna(avg_age) else "N/A")
        with kpi3:
            st.metric("Ciudad Más Común", most_common_city)

        st.subheader("Visualizaciones Detalladas")

        if not df_display.empty:
            # Distribución de Edad (Box Plot y Violin Plot)
            st.markdown("#### Distribución de Edad")
            fig_age, axes_age = plt.subplots(1, 2, figsize=(16, 6))
            sns.boxplot(y=df_display['edad'].dropna(), ax=axes_age[0], color="skyblue")
            axes_age[0].set_title('Diagrama de Caja de Edad')
            axes_age[0].set_ylabel('Edad')
            sns.violinplot(y=df_display['edad'].dropna(), ax=axes_age[1], color="lightgreen")
            axes_age[1].set_title('Diagrama de Violín de Edad')
            axes_age[1].set_ylabel('Edad')
            st.pyplot(fig_age)
            st.markdown("Los diagramas de caja y violín ayudan a visualizar la distribución de la edad y la presencia de valores atípicos.")

            # Distribución de Género por Ciudad
            st.markdown("#### Distribución de Pacientes por Género y Ciudad")
            # Asegurarse de que haya datos para agrupar
            if not df_display[['ciudad', 'sexo']].dropna().empty:
                sex_city_counts = df_display.groupby(['ciudad', 'sexo']).size().unstack(fill_value=0)
                fig_sex_city, ax_sex_city = plt.subplots(figsize=(12, 7))
                sex_city_counts.plot(kind='bar', stacked=True, ax=ax_sex_city, cmap='Pastel1')
                ax_sex_city.set_title('Número de Pacientes por Ciudad y Género')
                ax_sex_city.set_xlabel('Ciudad')
                ax_sex_city.set_ylabel('Número de Pacientes')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig_sex_city)
                st.dataframe(sex_city_counts)
                st.markdown("Este gráfico de barras apiladas muestra la composición por género dentro de cada ciudad.")
            else:
                st.info("No hay datos suficientes para generar el gráfico de Género por Ciudad con los filtros actuales.")

            # Edad Promedio por Ciudad y Género
            st.markdown("#### Edad Promedio por Ciudad y Género")
            if not df_display[['ciudad', 'sexo', 'edad']].dropna().empty:
                avg_age_city_sex = df_display.groupby(['ciudad', 'sexo'])['edad'].mean().unstack()
                
                # --- FIX: Ensure numerical type and replace any non-numeric with NaN ---
                # Convert the DataFrame to float type, coercing any non-numeric to NaN
                # Then fill any explicit None or NA with np.nan for heatmap compatibility
                avg_age_city_sex = avg_age_city_sex.astype(float).fillna(np.nan)
                # --- END FIX ---

                fig_avg_age, ax_avg_age = plt.subplots(figsize=(12, 7))
                sns.heatmap(avg_age_city_sex, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, ax=ax_avg_age)
                ax_avg_age.set_title('Edad Promedio por Ciudad y Género')
                st.pyplot(fig_avg_age)
                st.dataframe(avg_age_city_sex)
                st.markdown("Un mapa de calor para visualizar rápidamente la edad promedio en diferentes combinaciones de ciudad y género.")
            else:
                st.info("No hay datos suficientes para generar el mapa de calor de Edad Promedio por Ciudad y Género con los filtros actuales.")

        else:
            st.info("No hay datos para mostrar con los filtros seleccionados.")

# --- Nueva Sección 5: Modelado de Machine Learning (Agrupación) ---
elif selected_section == "5. Modelado de Machine Learning":
    st.header("5. 🧠 Modelado de Machine Learning: Agrupación de Pacientes (Clustering)")
    st.markdown("Identificación de segmentos de pacientes con características similares utilizando K-Means.")

    if 'df_cleaned' not in st.session_state:
        st.warning("Por favor, navega primero a la sección 'Limpieza y Validación' para cargar los datos limpios.")
    else:
        df_ml = st.session_state['df_cleaned'].copy()

        st.subheader("Preparación de Datos para ML")
        # Seleccionar características numéricas para clustering
        features = ['edad'] # Por ahora, solo edad. Si tienes más, añádelas aquí.

        # Eliminar filas con nulos en las características seleccionadas (para clustering)
        # Convertir 'edad' a flotante para manejar NaN si aún quedan (KMeans no los acepta directamente)
        df_ml['edad'] = df_ml['edad'].astype(float)
        df_ml_filtered = df_ml.dropna(subset=features)

        if df_ml_filtered.empty:
            st.warning("No hay suficientes datos limpios y completos para realizar el clustering con las características seleccionadas.")
        else:
            X = df_ml_filtered[features]

            # Escalado de características
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            st.write("Datos escalados para el modelo de clustering (primeras 5 filas):")
            st.dataframe(pd.DataFrame(X_scaled, columns=features).head())
            st.markdown("**Justificación:** El escalado es crucial para algoritmos basados en distancia como K-Means, asegurando que ninguna característica domine debido a su escala.")

            st.subheader("Determinación del Número Óptimo de Clusters (Método del Codo)")

            # --- Implementación Manual del Método del Codo ---
            sse = [] # Suma de Errores Cuadrados (o Inercia)
            # Prueba un rango de K de 1 a 10 (o ajusta según sea necesario)
            k_range = range(1, 11)

            for k in k_range:
                try:
                    # n_init='auto' es el valor recomendado para KMeans modernos
                    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    kmeans_model.fit(X_scaled)
                    sse.append(kmeans_model.inertia_)
                except ValueError as e:
                    # Capturar error si k=1 y hay solo una característica (ej., 'edad')
                    # Esto puede ocurrir con KMeans para n_clusters=1 en algunas versiones de sklearn
                    if k == 1:
                        sse.append(0) # La inercia es 0 para 1 cluster si todos los puntos son el centro
                    else:
                        st.error(f"Error al calcular la inercia para k={k}: {e}")
                        sse.append(None) # Añadir None si ocurre un error para otros k

            # Graficar el Método del Codo
            fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
            ax_elbow.plot(k_range, sse, marker='o', linestyle='--')
            ax_elbow.set_title('Método del Codo para K-Means')
            ax_elbow.set_xlabel('Número de Clusters (k)')
            ax_elbow.set_ylabel('Inercia (SSE)')
            ax_elbow.grid(True)
            st.pyplot(fig_elbow)
            st.markdown("""
            El **Método del Codo** ayuda a determinar el número óptimo de clusters (`k`). Se busca el punto en el gráfico donde la inercia (suma de cuadrados dentro del cluster) disminuye significativamente, formando una "rodilla" o "codo".
            """)
            # --- Fin de la Implementación Manual del Método del Codo ---

            # Slider para que el usuario elija el número de clusters
            st.subheader("Configuración del Modelo K-Means")
            n_clusters = st.slider("Selecciona el número de clusters (k):", min_value=2, max_value=8, value=3)

            # Entrenamiento del modelo
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            df_ml_filtered['cluster'] = kmeans.fit_predict(X_scaled)
            st.success(f"Modelo K-Means entrenado con **{n_clusters}** clusters.")

            st.subheader("Resultados del Agrupamiento")

            # Características promedio por cluster
            cluster_centers_scaled = kmeans.cluster_centers_
            cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled) # Volver a la escala original
            cluster_df = pd.DataFrame(cluster_centers_original, columns=features)
            cluster_df['Cluster'] = range(n_clusters)
            st.markdown("#### Características Promedio por Cluster (en escala original)")
            st.dataframe(cluster_df.set_index('Cluster'))
            st.markdown("Estos valores representan el centro de cada cluster, ayudando a interpretar lo que define a cada grupo de pacientes.")

            # Conteo de pacientes por cluster
            st.markdown("#### Conteo de Pacientes por Cluster")
            cluster_counts = df_ml_filtered['cluster'].value_counts().sort_index()
            fig_cluster_counts, ax_cluster_counts = plt.subplots(figsize=(8, 5))
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax_cluster_counts, palette="viridis")
            ax_cluster_counts.set_title('Número de Pacientes por Cluster')
            ax_cluster_counts.set_xlabel('Cluster')
            ax_cluster_counts.set_ylabel('Conteo de Pacientes')
            st.pyplot(fig_cluster_counts)
            st.dataframe(cluster_counts.to_frame(name='Conteo'))
            st.markdown("Este gráfico de barras apiladas muestra cuántos pacientes fueron asignados a cada cluster.")

            # Visualización de los clusters (si solo tenemos una característica como 'edad')
            st.markdown("#### Visualización de Clusters (Distribución de Edad por Cluster)")
            fig_cluster_dist, ax_cluster_dist = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df_ml_filtered, x='edad', hue='cluster', kde=True, palette='tab10', ax=ax_cluster_dist, bins=15)
            ax_cluster_dist.set_title('Distribución de Edad por Cluster')
            ax_cluster_dist.set_xlabel('Edad')
            ax_cluster_dist.set_ylabel('Frecuencia')
            st.pyplot(fig_cluster_dist)
            st.markdown("Este histograma superpuesto muestra cómo se distribuyen las edades dentro de cada cluster, ayudando a entender los perfiles de edad de cada grupo.")

            st.markdown("### **Interpretación y Aplicaciones:**")
            st.markdown(f"""
            Basado en la `edad`, el modelo K-Means ha identificado **{n_clusters}** grupos distintos de pacientes.
            Por ejemplo, si los clusters son:
            * **Cluster 0:** Podría representar pacientes jóvenes (ej. edad promedio de 20-30 años).
            * **Cluster 1:** Podría representar pacientes de mediana edad (ej. edad promedio de 40-50 años).
            * **Cluster 2:** Podría representar pacientes mayores (ej. edad promedio de 60+ años).

            **Aplicaciones potenciales:**
            * **Marketing y Comunicación Personalizada:** Enviar información relevante sobre prevención o programas de salud específicos para cada grupo de edad.
            * **Gestión de Recursos Hospitalarios:** Anticipar las necesidades de ciertos grupos de edad (ej., especialidades pediátricas para el cluster joven, geriatría para el cluster mayor).
            * **Investigación Clínica:** Estudiar patrones de enfermedades o tratamientos que sean más prevalentes en un segmento de edad particular.
            """)
