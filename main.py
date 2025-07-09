import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Análisis de Calidad de Datos Hospital")
st.title("🏥 Análisis y Calidad de Datos de Pacientes de Hospital")
st.markdown("Esta aplicación realiza un análisis exhaustivo de la calidad de los datos de pacientes, seguido de procesos de limpieza, validación y la generación de indicadores.")

# URL del archivo JSON (ajusta si tienes más archivos)
DATA_URL_PACIENTES = "https://raw.githubusercontent.com/JulianTorrest/hospital/refs/heads/main/dataset_hospital%202.json"

# --- Funciones Auxiliares para Carga de Datos y Cacheo ---
@st.cache_data
def load_data(url, key_name):
    """Carga los datos desde una URL y normaliza un JSON si tiene una clave raíz."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if key_name in data:
            df = pd.json_normalize(data[key_name])
        else:
            df = pd.DataFrame(data) # Asumir que es una lista de objetos directamente
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error de red al cargar los datos desde {url}: {e}")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"Error al decodificar JSON desde {url}: {e}")
        return pd.DataFrame()
    except KeyError:
        st.error(f"La clave '{key_name}' no se encontró en el JSON de {url}. Asegúrate de que el JSON tiene una estructura con '{key_name}'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al procesar los datos de {url}: {e}")
        return pd.DataFrame()

# Función para calcular la edad (usada en limpieza y validación)
def calculate_age_from_dob(row_dob, current_date):
    if pd.isna(row_dob):
        return None
    else:
        age = current_date.year - row_dob.year - ((current_date.month, current_date.day) < (row_dob.month, row_dob.day))
        return age if age >= 0 else None # Edad no puede ser negativa

# Cargar los datos de pacientes
df_pacientes = load_data(DATA_URL_PACIENTES, 'pacientes')

# --- Sidebar para Navegación ---
st.sidebar.header("Navegación")
selected_section = st.sidebar.radio(
    "Ir a la sección:",
    ("1. Exploración Inicial", "2. Limpieza y Validación", "3. Indicadores y Documentación")
)

# --- Contenido Principal de la Aplicación ---

# Sección 1: Análisis de Calidad de Datos (Exploración)
if selected_section == "1. Exploración Inicial":
    st.header("1. 📉 Análisis de Calidad de Datos (Exploración)")
    st.markdown("Identificación de los principales problemas de calidad en la tabla de pacientes.")

    if df_pacientes.empty:
        st.warning("No se pudieron cargar los datos de pacientes o el DataFrame está vacío.")
    else:
        st.subheader("1.1. Vista Previa de los Datos Originales")
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
            - **`fecha_nacimiento`**: Aunque no hay nulos directos, es importante verificar el formato y la validez de las fechas.
            """)
        else:
            st.info("No se detectaron valores faltantes significativos en los datos cargados.")

        st.subheader("1.4. Inconsistencias y Formatos")

        st.markdown("#### Columna `sexo`")
        st.write(df_pacientes['sexo'].value_counts(dropna=False))
        # Convertir a minúsculas para verificar inconsistencias más fácilmente
        if len(df_pacientes['sexo'].astype(str).str.lower().unique()) > 2 and 'female' in df_pacientes['sexo'].astype(str).str.lower().unique() and 'male' in df_pacientes['sexo'].astype(str).str.lower().unique():
            st.warning("Problema: Inconsistencia en el uso de mayúsculas/minúsculas o variaciones en la columna `sexo` (Ej: 'Female' vs 'female', o otros valores inesperados).")
        elif 'Female' not in df_pacientes['sexo'].unique().tolist() and 'Male' not in df_pacientes['sexo'].unique().tolist():
            st.warning("Problema: Los valores de `sexo` no están estandarizados a 'Female' y 'Male'.")


        st.markdown("#### Columna `fecha_nacimiento`")
        # Verificar formatos no válidos de fecha
        invalid_dates = df_pacientes[pd.to_datetime(df_pacientes['fecha_nacimiento'], errors='coerce').isna() & df_pacientes['fecha_nacimiento'].notna()]
        if not invalid_dates.empty:
            st.warning(f"Problema: Se encontraron {len(invalid_dates)} fechas de nacimiento con formato inválido.")
            st.dataframe(invalid_dates)
        else:
            st.info("No se encontraron fechas de nacimiento con formato inválido aparente.")

        st.markdown("#### Columna `email`")
        # Validación básica de email
        invalid_emails = df_pacientes[~df_pacientes['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        if not invalid_emails.empty:
            st.warning(f"Problema: Se encontraron {len(invalid_emails)} emails con formato potencialmente inválido.")
            st.dataframe(invalid_emails.head())
        else:
            st.info("No se encontraron emails con formato inválido aparente (validación básica).")

        st.markdown("#### Columna `telefono`")
        # Validación básica de teléfono (solo si no es nulo y no es un número tras limpiar no-dígitos)
        # Primero, intentar limpiar para verificar si quedan problemas.
        temp_telefono_cleaned = df_pacientes['telefono'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        non_numeric_phones_after_temp_clean = df_pacientes[temp_telefono_cleaned.notna() & ~temp_telefono_cleaned.str.isdigit()]
        if not non_numeric_phones_after_temp_clean.empty:
            st.warning(f"Problema: Se encontraron {len(non_numeric_phones_after_temp_clean)} números de teléfono con caracteres no numéricos o que no se convierten a un formato numérico válido.")
            st.dataframe(non_numeric_phones_after_temp_clean.head())
        else:
            st.info("No se encontraron números de teléfono con caracteres no numéricos aparentes.")


        st.markdown("""
        ### **Resumen de Problemas de Calidad (Pacientes):**

        1.  **Valores Nulos:** Principalmente en la columna `edad`.
        2.  **Inconsistencias de Formato:**
            * `sexo`: Posibles variaciones en mayúsculas/minúsculas (`Female` vs `female`).
            * `fecha_nacimiento`: Necesita conversión a tipo `datetime` y manejo de posibles formatos incorrectos.
            * `edad`: Debe ser un valor numérico y coherente con `fecha_nacimiento`. Si es `null`, debe calcularse.
            * `email`, `telefono`: Requieren validación de formato (aunque el ejemplo dado parece limpio, es una buena práctica).
        3.  **Coherencia de Datos:** La `edad` debe ser derivable de `fecha_nacimiento` y ser un número positivo.
        """)
        st.info("Nota: Dado que solo tenemos la tabla de 'pacientes' de la URL, el análisis se centra en ella.")


# Sección 2: Limpieza y Validación
elif selected_section == "2. Limpieza y Validación":
    st.header("2. 🧹 Limpieza y Validación")
    st.markdown("Aplicación de un proceso de limpieza para resolver los problemas identificados y validaciones cruzadas.")

    if df_pacientes.empty:
        st.warning("No se pudieron cargar los datos de pacientes para la limpieza.")
    else:
        df_cleaned = df_pacientes.copy() # Trabajar en una copia para no modificar el original
        current_date = date.today() # Definir la fecha actual una sola vez

        # --- Limpieza de Datos ---
        st.subheader("2.1. Proceso de Limpieza")

        st.markdown("#### Limpieza de `sexo`")
        st.code("""df_cleaned['sexo'] = df_cleaned['sexo'].astype(str).str.capitalize() # Normalizar a 'Female' o 'Male'""")
        df_cleaned['sexo'] = df_cleaned['sexo'].astype(str).str.capitalize()
        # Manejar posibles valores 'nan' convertidos a 'Nan'
        df_cleaned.loc[df_cleaned['sexo'] == 'Nan', 'sexo'] = None
        st.write("Valores de `sexo` después de la normalización:")
        st.write(df_cleaned['sexo'].value_counts(dropna=False))
        st.markdown("**Justificación:** Se normaliza el texto a `Capitalize` para estandarizar 'Female'/'Male' y evitar inconsistencias por mayúsculas/minúsculas. Los valores nulos convertidos a 'Nan' se vuelven a establecer como `None`.")

        st.markdown("#### Limpieza y Cálculo de `fecha_nacimiento` y `edad`")
        st.code("""
# Convertir 'fecha_nacimiento' a datetime, forzando nulos si el formato es inválido
df_cleaned['fecha_nacimiento'] = pd.to_datetime(df_cleaned['fecha_nacimiento'], errors='coerce')

# Calcular 'edad' para nulos o valores inconsistentes
current_date = date.today() # Fecha actual
df_cleaned['edad_calculada'] = df_cleaned['fecha_nacimiento'].apply(lambda dob: calculate_age_from_dob(dob, current_date))

# Priorizar la edad calculada si fecha_nacimiento es válida, sino usar la existente o None
df_cleaned['edad'] = df_cleaned.apply(
    lambda row: row['edad_calculada'] if pd.notna(row['edad_calculada']) else row['edad'], axis=1
)
df_cleaned['edad'] = df_cleaned['edad'].astype('Int64') # Int64 para permitir NaNs y que sea entero
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
        - Se convierte `fecha_nacimiento` a tipo `datetime`, convirtiendo los formatos inválidos a `NaT` (Not a Time).
        - Se recalcula la `edad` en base a `fecha_nacimiento` si esta es válida. Se prioriza esta edad calculada si está disponible. Si `fecha_nacimiento` es `NaT`, se mantiene la `edad` original.
        - Se asegura que la edad sea un número entero no negativo. Se usa `Int64` para manejar nulos en columnas numéricas.
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
        st.markdown("**Justificación:** Se eliminan caracteres no numéricos del teléfono para estandarizar el formato. Las cadenas vacías o solo con espacios resultantes se convierten a `None`.")

        st.subheader("2.2. Validaciones Cruzadas entre Campos")
        st.markdown("Se aplican reglas para asegurar la coherencia lógica entre las columnas.")

        st.markdown("#### Validación: `edad` coherente con `fecha_nacimiento`")
        # Recalcular la edad para comparar con la edad final limpia
        df_cleaned_temp_age_check = df_cleaned.copy()
        df_cleaned_temp_age_check['calculated_age_for_check'] = df_cleaned_temp_age_check['fecha_nacimiento'].apply(lambda dob: calculate_age_from_dob(dob, current_date))

        inconsistent_ages = df_cleaned[
            (df_cleaned['edad'].notna()) &
            (df_cleaned_temp_age_check['calculated_age_for_check'].notna()) &
            (abs(df_cleaned['edad'] - df_cleaned_temp_age_check['calculated_age_for_check']) > 1) # Tolerancia de 1 año por posibles desfases de actualización
        ]
        if not inconsistent_ages.empty:
            st.warning(f"Se encontraron {len(inconsistent_ages)} registros con **edad inconsistente** con la fecha de nacimiento (diferencia > 1 año) *después de la limpieza*.")
            st.dataframe(inconsistent_ages[['id_paciente', 'fecha_nacimiento', 'edad']].head())
            st.markdown("""
            **Regla de Validación:** La edad calculada a partir de `fecha_nacimiento` debe ser consistente con la `edad` reportada (se permite una pequeña tolerancia para desfases de fecha de actualización).
            **Acción:** La limpieza ya prioriza la edad calculada si `fecha_nacimiento` es válida, minimizando estas inconsistencias. Si aún existen, podría indicar una `fecha_nacimiento` errónea.
            """)
        else:
            st.success("No se encontraron inconsistencias significativas entre `edad` y `fecha_nacimiento` después de la limpieza.")

        st.markdown("#### Validación: `email` con formato válido")
        invalid_email_after_clean = df_cleaned[~df_cleaned['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        if not invalid_email_after_clean.empty:
            st.warning(f"Se encontraron {len(invalid_email_after_clean)} registros con **email inválido** después de la limpieza.")
            st.dataframe(invalid_email_after_clean[['id_paciente', 'email']].head())
            st.markdown("""
            **Regla de Validación:** El campo `email` debe seguir un formato estándar de correo electrónico (`texto@texto.dominio`).
            **Acción:** Se identifican, pero no se modifican automáticamente ya que requeriría inferencia o interacción manual.
            """)
        else:
            st.success("Todos los emails parecen tener un formato válido después de la limpieza (validación básica).")

        st.markdown("#### Validación: `telefono` solo contiene dígitos (tras limpieza)")
        non_numeric_phones_cleaned = df_cleaned[df_cleaned['telefono'].notna() & ~df_cleaned['telefono'].astype(str).str.isdigit()]
        if not non_numeric_phones_cleaned.empty:
            st.warning(f"Se encontraron {len(non_numeric_phones_cleaned)} registros con **teléfonos con caracteres no numéricos** después de la limpieza (esto no debería ocurrir si la limpieza fue efectiva).")
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

        # Guardar el DataFrame limpio y original en el estado de la sesión para usarlo en la siguiente sección
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

        st.markdown("#### Comparativa de Valores Nulos (%)")
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
        - Se espera una **reducción significativa** en el porcentaje de nulos en `edad` si `fecha_nacimiento` estaba disponible y válida.
        - `fecha_nacimiento` puede mostrar un aumento de nulos si los formatos originales eran inválidos y se convirtieron a `NaT`.
        - `telefono` puede tener nulos si quedaron cadenas vacías tras la limpieza de no numéricos.
        - `sexo` podría tener nulos si había valores vacíos o no estandarizables.
        """)

        st.markdown("#### Comparativa de Tipos de Datos")
        st.write("Tipos de datos originales:")
        st.json(indicators_original['Tipos de Datos por Columna'])
        st.write("Tipos de datos después de la limpieza:")
        st.json(indicators_cleaned['Tipos de Datos por Columna'])
        st.markdown("""
        **Observaciones:**
        - `fecha_nacimiento` debe pasar de `object` (cadena de texto) a `datetime64[ns]` (tipo fecha y hora).
        - `edad` debe pasar de `object` (si contenía nulos o era mixto) o `float64` (si se infirió numérico) a `Int64` (entero con soporte para nulos).
        - `telefono` y `email` idealmente deberían permanecer como `object` (cadena) pero con formato validado.
        """)

        st.markdown("#### Indicadores de Consistencia y Unicidad")
        st.write("**`sexo` - Unicidad de Categorías:**")
        st.write(f"Original: {df_original['sexo'].value_counts(dropna=False).index.tolist()}")
        st.write(f"Limpio: {df_cleaned['sexo'].value_counts(dropna=False).index.tolist()}")
        st.markdown("""
        **Observación:** Se espera que el número de categorías únicas y sus nombres se normalicen después de la limpieza (ej: solo 'Female', 'Male' y `None`).
        """)

        st.write("**`email` - Patrón de Formato (Conteo de Inválidos):**")
        invalid_emails_original = df_original[~df_original['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        invalid_emails_cleaned = df_cleaned[~df_cleaned['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        st.write(f"Emails inválidos (Original): {len(invalid_emails_original)}")
        st.write(f"Emails inválidos (Limpio): {len(invalid_emails_cleaned)}")
        st.markdown("""
        **Observación:** Aunque la limpieza no los altera, se validó su formato. Este indicador muestra si persisten emails con formato no estándar.
        """)

        st.subheader("3.2. Documentación del Proceso")

        st.markdown("### **Supuestos Adoptados Durante la Limpieza:**")
        st.markdown("""
        * **Fuente Única para Edad:** Se asume que `fecha_nacimiento` es la fuente más confiable para determinar la `edad`. Si `fecha_nacimiento` es válida, se **prioriza el cálculo de la edad a partir de ella** sobre el valor existente de `edad` si este es nulo o inconsistente. La edad se calcula como la diferencia de años a la fecha actual, ajustando por mes y día.
        * **Formato de `sexo`:** Se asume que los valores `Female`, `female`, `Male`, `male` y sus variaciones deben ser estandarizados a `Female` y `Male` (capitalización de la primera letra). Otros valores (`NaN`, vacíos) se mantienen como `None`.
        * **Formato de `telefono`:** Se asume que los números de teléfono solo deben contener dígitos. Cualquier otro carácter (guiones, espacios, paréntesis, etc.) se **elimina**. Las cadenas vacías o compuestas solo por espacios resultantes de esta limpieza se interpretan como nulas (`None`).
        * **Coherencia de Fechas:** Se asume que las fechas de nacimiento no pueden estar en el futuro ni ser extremadamente antiguas (se calcula la edad relativa a la fecha actual y se descartan edades negativas, convirtiéndolas a `None`).
        * **ID de Paciente:** Se asume que `id_paciente` es el identificador **único** para cada paciente y no se espera que tenga problemas de calidad (duplicados, nulos).
        """)

        st.markdown("### **Reglas de Validación Implementadas:**")
        st.markdown("""
        * **Validación de `fecha_nacimiento`:** Se verifica que la columna pueda ser convertida al tipo `datetime`. Los valores que no cumplan este formato se marcan como `NaT` (Not a Time).
        * **Validación de `edad`:**
            * Debe ser un número entero no negativo.
            * Debe ser **coherente** con `fecha_nacimiento`: la `edad` calculada a partir de `fecha_nacimiento` debe ser cercana a la `edad` reportada (se permite una tolerancia de 1 año para posibles desfases de fecha de actualización en los datos originales).
        * **Validación de `sexo`:** Los valores deben estar dentro de un conjunto predefinido de categorías estandarizadas (`Female`, `Male`, o `None`).
        * **Validación de `email`:** Se verifica que el formato siga una expresión regular básica (`[^@]+@[^@]+\.[^@]+`) para asegurar que contenga un `@` y al menos un `.` en el dominio. Esto es una validación de patrón, no de existencia.
        * **Validación de `telefono`:** Se verifica que, después de la limpieza, la columna solo contenga caracteres numéricos (o sea nula).
        """)

        st.markdown("### **Recomendaciones de Mejora para Asegurar la Calidad Futura de los Datos:**")
        st.markdown("""
        1.  **Validación en la Fuente:** Implementar validaciones a nivel de entrada de datos (ej., formularios web, bases de datos) para `fecha_nacimiento`, `sexo`, `email` y `telefono`.
            * **`fecha_nacimiento`:** Usar selectores de fecha para evitar entradas manuales erróneas y asegurar el formato `YYYY-MM-DD`.
            * **`sexo`:** Usar menús desplegables (`dropdowns`) con opciones predefinidas (`Female`, `Male`) para evitar inconsistencias de capitalización o errores tipográficos.
            * **`email`:** Implementar validación de formato de email en tiempo real al ingresar los datos y, si es posible, una verificación de dominio.
            * **`telefono`:** Forzar la entrada de solo dígitos o un formato específico (ej., con máscaras de entrada) según el país, y validar la longitud mínima/máxima.
        2.  **Estandarización de `ciudad`:** Implementar un catálogo o lista maestra de ciudades/municipios para asegurar la consistencia y evitar variaciones en nombres de ciudades (ej., "Barranquilla" vs "barranquilla", o errores tipográficos).
        3.  **Definición de Campos Obligatorios:** Establecer claramente qué campos son obligatorios (ej., `id_paciente`, `nombre`, `fecha_nacimiento`, `sexo`) en la base de datos o sistema de entrada para reducir la aparición de valores nulos críticos.
        4.  **Auditorías Regulares de Datos:** Realizar auditorías periódicas de la base de datos para identificar nuevos patrones de errores o degradación de la calidad de los datos a lo largo del tiempo.
        5.  **Documentación de Metadatos:** Mantener un diccionario de datos (`data dictionary`) actualizado que defina claramente cada campo, su tipo de dato esperado, formato, reglas de validación y significado, accesible para todo el equipo.
        6.  **Sistema de Reporte de Errores:** Establecer un mecanismo para que los usuarios (personal del hospital, médicos) puedan reportar inconsistencias o errores de datos cuando los detecten, con un flujo claro para su corrección.
        7.  **Capacitación del Personal:** Asegurar que el personal que ingresa los datos esté continuamente capacitado en las mejores prácticas de entrada de datos y entienda la importancia de la calidad de los datos para la toma de decisiones y la atención al paciente.
        """)

        st.subheader("3.3. Bonus (Opcional)")
        st.markdown("""
        #### Implementación de Pruebas Automáticas
        Para implementar pruebas automáticas para la calidad de los datos, se podrían usar frameworks como **Pytest** o **Great Expectations**.

        **Ejemplo conceptual con Pytest (en un archivo `tests/test_data_quality.py`):**
        ```python
        # Este código es conceptual y no forma parte de app.py
        # Deberías tener tus funciones de limpieza y validación en un módulo separado para poder importarlas aquí.
        import pandas as pd
        import pytest
        # from your_project.data_quality_functions import clean_patient_data, validate_email_format # Ejemplo de importación

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
                    {"id_paciente": 6, "nombre": "Carlos", "fecha_nacimiento": "1970-07-08", "edad": 50, "sexo": "OTHER", "email": "carlos@example.com", "telefono": "123-456-7890", "ciudad": "Cartagena"}
                ]
            }
            return pd.json_normalize(data['pacientes'])

        def test_sexo_standardization(sample_patient_data):
            # Asumiendo que df_cleaned es el resultado de tu función de limpieza
            df_cleaned = sample_patient_data.copy() # Reemplaza con la llamada a tu función de limpieza real
            df_cleaned['sexo'] = df_cleaned['sexo'].astype(str).str.capitalize()
            df_cleaned.loc[df_cleaned['sexo'] == 'Nan', 'sexo'] = None
            df_cleaned.loc[~df_cleaned['sexo'].isin(['Female', 'Male', None]), 'sexo'] = None # Normalizar 'OTHER' a None
            assert all(s in ['Female', 'Male', None] for s in df_cleaned['sexo'].unique()), "Los valores de sexo no están estandarizados o contienen inesperados."

        def test_age_calculation_and_validation(sample_patient_data):
            df_cleaned = sample_patient_data.copy() # Reemplaza con la llamada a tu función de limpieza real
            # Simular la lógica de limpieza de edad
            df_cleaned['fecha_nacimiento'] = pd.to_datetime(df_cleaned['fecha_nacimiento'], errors='coerce')
            current_date_for_test = date(2025, 7, 8) # Fecha fija para pruebas
            df_cleaned['edad_calculada_test'] = df_cleaned['fecha_nacimiento'].apply(lambda dob: calculate_age_from_dob(dob, current_date_for_test))
            df_cleaned['edad_final'] = df_cleaned.apply(
                lambda row: row['edad_calculada_test'] if pd.notna(row['edad_calculada_test']) else row['edad'], axis=1
            )
            df_cleaned['edad_final'] = df_cleaned['edad_final'].astype('Int64')

            # Verificar que no hay edades negativas
            assert all(df_cleaned['edad_final'].dropna() >= 0), "Las edades calculadas no deben ser negativas."
            # Verificar que la edad para id_paciente 1 fue calculada (1954-01-08)
            assert df_cleaned.loc[df_cleaned['id_paciente'] == 1, 'edad_final'].iloc[0] == 71, "La edad para paciente 1 no se calculó correctamente."
            # Verificar que la fecha futura resulta en edad nula/None
            assert pd.isna(df_cleaned.loc[df_cleaned['id_paciente'] == 3, 'edad_final'].iloc[0]), "Edad para fecha futura debería ser nula."


        def test_email_format_after_cleaning(sample_patient_data):
            df_cleaned = sample_patient_data.copy() # No hay limpieza directa de email, solo validación
            invalid_emails = df_cleaned[~df_cleaned['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
            # Esperamos que 'pedro@example' siga siendo inválido
            assert 'pedro@example' in invalid_emails['email'].values, "Email 'pedro@example' no fue marcado como inválido."
            assert len(invalid_emails) > 0, "No se detectaron emails inválidos conocidos."


        def test_telefono_numeric_after_cleaning(sample_patient_data):
            df_cleaned = sample_patient_data.copy() # Reemplaza con la llamada a tu función de limpieza real
            df_cleaned['telefono_limpio_test'] = df_cleaned['telefono'].astype(str).str.replace(r'[^0-9]', '', regex=True)
            df_cleaned.loc[df_cleaned['telefono_limpio_test'].str.strip() == '', 'telefono_limpio_test'] = None
            assert all(df_cleaned['telefono_limpio_test'].dropna().apply(lambda x: x.isdigit())), "El campo teléfono contiene caracteres no numéricos después de la limpieza."
            assert pd.isna(df_cleaned.loc[df_cleaned['id_paciente'] == 2, 'telefono_limpio_test'].iloc[0]), "Teléfono con caracteres no numéricos no se limpió correctamente."

        ```
        Para ejecutar Pytest, necesitas:
        1.  Tener instalada la librería `pytest`: `pip install pytest`
        2.  Guardar el código de prueba en un archivo `tests/test_data_quality.py` (o similar).
        3.  Refactorizar tus funciones de limpieza y validación de `app.py` en un módulo Python separado (ej. `data_processing.py`) para poder importarlas en los tests.
        4.  Ejecutar `pytest` en tu terminal desde la raíz de tu proyecto.

        #### Simulación de Migración de Datos Limpios a una Estructura Destino
        Una vez que los datos han sido limpiados y validados, el siguiente paso lógico en un pipeline de datos es cargarlos en una estructura destino, como un Data Warehouse o una base de datos analítica. Los formatos como **Parquet** son ideales para esto debido a su naturaleza columnar, compresión eficiente y capacidad para manejar esquemas complejos.

        Aquí simulamos la descarga de los datos limpios en formatos comunes para migración.
        """)

        # Código para el bonus de descarga (añadir esto dentro del 'if selected_section == "3. Indicadores y Documentación":' block)
        if 'df_cleaned' in st.session_state:
            st.markdown("#### Simulación de Migración de Datos Limpios")

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            @st.cache_data
            def convert_df_to_parquet(df):
                try:
                    import pyarrow # Asegurarse de que pyarrow esté instalado para to_parquet
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
                if parquet_data:
                    st.download_button(
                        label="Descargar Datos Limpios (Parquet)",
                        data=parquet_data,
                        file_name="pacientes_limpios.parquet",
                        mime="application/octet-stream",
                        key="download_parquet"
                    )

            st.markdown("""
            **Justificación de Migración:**
            La migración de datos limpios a un Data Warehouse (DW) implica típicamente:
            1.  **Extracción (Extract):** Obtener los datos de las fuentes.
            2.  **Transformación (Transform):** Los datos se limpian, estandarizan, validan y se preparan para adaptarse al esquema del DW. Esta es la fase que hemos detallado en esta aplicación.
            3.  **Carga (Load):** Los datos transformados se cargan en las tablas dimensionales y de hechos del DW.
            Los formatos como CSV son universales, pero Parquet es preferido en entornos de Big Data y DW por su eficiencia. La simulación de descarga en CSV/Parquet representa la salida de este proceso de transformación listo para ser cargado en un sistema destino optimizado para consultas analíticas.
            """)
