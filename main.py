import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Initial Configuration ---
st.set_page_config(layout="wide", page_title="Análisis de Calidad de Datos Hospital")
st.title("🏥 Análisis y Calidad de Datos de Pacientes de Hospital")
st.markdown("This app performs a comprehensive analysis of patient data quality, followed by cleaning, validation processes, KPI generation, advanced EDA, and a Machine Learning model.")

# URL of the JSON file
DATA_URL_PACIENTES = "https://raw.githubusercontent.com/JulianTorrest/hospital/refs/heads/main/dataset_hospital%202.json"

# --- Helper Functions for Data Loading and Caching ---
@st.cache_data
def load_data(url, key_name):
    """Loads data from a URL and normalizes a JSON if it has a root key."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if key_name in data:
            df = pd.json_normalize(data[key_name])
        else:
            df = pd.DataFrame(data) # Assume it's a list of objects directly
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Network error loading data from {url}: {e}")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"Error decoding JSON from {url}: {e}")
        return pd.DataFrame()
    except KeyError:
        st.error(f"The key '{key_name}' was not found in the JSON from {url}. Ensure the JSON has a structure with '{key_name}'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred processing data from {url}: {e}")
        return pd.DataFrame()

# Function to calculate age (used in cleaning and validation)
def calculate_age_from_dob(row_dob, current_date):
    if pd.isna(row_dob):
        return None
    else:
        # Ensure row_dob is a datetime object, not just a string
        if isinstance(row_dob, str):
            try:
                row_dob = datetime.strptime(row_dob, '%Y-%m-%d').date()
            except ValueError:
                return None # Could not parse the date
        elif isinstance(row_dob, pd.Timestamp):
            row_dob = row_dob.date()

        age = current_date.year - row_dob.year - ((current_date.month, current_date.day) < (row_dob.month, row_dob.day))
        return age if age >= 0 else None # Age cannot be negative

# Load patient data
df_pacientes = load_data(DATA_URL_PACIENTES, 'pacientes')

# --- Sidebar for Navigation ---
st.sidebar.header("Navegación")
selected_section = st.sidebar.radio(
    "Ir a la sección:",
    ("1. Exploración Inicial", "2. Limpieza y Validación", "3. Indicadores y Documentación", "4. EDA Avanzado & Dashboards", "5. Modelado de Machine Learning")
)

# --- Main Application Content ---

# Section 1: Data Quality Analysis (Exploration)
if selected_section == "1. Exploración Inicial":
    st.header("1. 📉 Análisis de Calidad de Datos (Exploración)")
    st.markdown("Identification of the main quality issues in the patient table.")

    if df_pacientes.empty:
        st.warning("Could not load patient data or DataFrame is empty.")
    else:
        st.subheader("1.1. Preview of Original Data")
        st.dataframe(df_pacientes.head())

        st.subheader("1.2. General Information and Data Types")
        buffer = pd.io.common.StringIO()
        df_pacientes.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("1.3. Missing Values (Nulls)")
        missing_values = df_pacientes.isnull().sum()
        missing_percentage = (df_pacientes.isnull().sum() / len(df_pacientes)) * 100
        missing_df = pd.DataFrame({
            'Valores Faltantes': missing_values,
            'Porcentaje (%)': missing_percentage
        }).sort_values(by='Porcentaje (%)', ascending=False)
        st.dataframe(missing_df[missing_df['Valores Faltantes'] > 0])

        if not missing_df[missing_df['Valores Faltantes'] > 0].empty:
            st.markdown("""
            **Initial Observations on Missing Values:**
            - **`edad`**: Shows `null` in the JSON for some records. This is a problem, as age is crucial and can be calculated from the date of birth.
            - **`fecha_nacimiento`**: Although there are no direct nulls, it's important to verify date format and validity.
            """)
        else:
            st.info("No significant missing values detected in the loaded data.")

        st.subheader("1.4. Inconsistencies and Formats")

        st.markdown("#### `sexo` Column")
        st.write(df_pacientes['sexo'].value_counts(dropna=False))
        # Convert to lowercase to check for inconsistencies more easily
        sex_lower_unique = df_pacientes['sexo'].astype(str).str.lower().unique()
        if len(sex_lower_unique) > 2 and 'female' in sex_lower_unique and 'male' in sex_lower_unique:
            st.warning("Problem: Inconsistency in capitalization or variations in the `sexo` column (e.g., 'Female' vs 'female', or other unexpected values).")
        elif 'Female' not in df_pacientes['sexo'].unique().tolist() and 'Male' not in df_pacientes['sexo'].unique().tolist():
            # This condition is stricter if you expect only exact 'Female'/'Male' before cleaning
            st.warning("Problem: `sexo` values are not standardized to 'Female' and 'Male' (considering capitalization).")


        st.markdown("#### `fecha_nacimiento` Column")
        # Check for invalid date formats
        invalid_dates = df_pacientes[pd.to_datetime(df_pacientes['fecha_nacimiento'], errors='coerce').isna() & df_pacientes['fecha_nacimiento'].notna()]
        if not invalid_dates.empty:
            st.warning(f"Problem: **{len(invalid_dates)}** birth dates with invalid format were found.")
            st.dataframe(invalid_dates)
        else:
            st.info("No apparent invalid birth date formats were found.")

        st.markdown("#### `email` Column")
        # Basic email validation
        invalid_emails = df_pacientes[~df_pacientes['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        if not invalid_emails.empty:
            st.warning(f"Problem: **{len(invalid_emails)}** emails with potentially invalid format were found.")
            st.dataframe(invalid_emails.head())
        else:
            st.info("No apparent invalid email formats were found (basic validation).")

        st.markdown("#### `telefono` Column")
        # Basic phone validation (only if not null and not a number after cleaning non-digits)
        # First, try to clean to check if problems remain.
        temp_telefono_cleaned = df_pacientes['telefono'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        non_numeric_phones_after_temp_clean = df_pacientes[temp_telefono_cleaned.notna() & ~temp_telefono_cleaned.str.isdigit()]
        if not non_numeric_phones_after_temp_clean.empty:
            st.warning(f"Problem: **{len(non_numeric_phones_after_temp_clean)}** phone numbers with non-numeric characters or that do not convert to a valid numeric format were found.")
            st.dataframe(non_numeric_phones_after_temp_clean.head())
        else:
            st.info("No apparent non-numeric phone numbers were found.")


        st.markdown("""
        ### **Summary of Quality Problems (Patients):**

        1.  **Null Values:** Mainly in the `edad` column.
        2.  **Format Inconsistencies:**
            * `sexo`: Possible variations in capitalization (`Female` vs `female`).
            * `fecha_nacimiento`: Needs conversion to `datetime` type and handling of possible incorrect formats.
            * `edad`: Must be a numeric value and consistent with `fecha_nacimiento`. If it's `null`, it should be calculated.
            * `email`, `telefono`: Require format validation (although the given example seems clean, it's good practice).
        3.  **Data Coherence:** `edad` must be derivable from `fecha_nacimiento` and be a positive number.
        """)
        st.info("Note: Since we only have the 'pacientes' table from the URL, the analysis focuses on it.")


# Section 2: Cleaning and Validation
elif selected_section == "2. Limpieza y Validación":
    st.header("2. 🧹 Limpieza y Validación")
    st.markdown("Application of a cleaning process to resolve identified problems and cross-validations.")

    if df_pacientes.empty:
        st.warning("Could not load patient data for cleaning.")
    else:
        df_cleaned = df_pacientes.copy() # Work on a copy to avoid modifying the original
        current_date = date.today() # Define the current date once

        # --- Data Cleaning ---
        st.subheader("2.1. Cleaning Process")

        st.markdown("#### Cleaning `sexo`")
        st.code("""df_cleaned['sexo'] = df_cleaned['sexo'].astype(str).str.capitalize() # Normalize to 'Female' or 'Male'
df_cleaned.loc[df_cleaned['sexo'] == 'Nan', 'sexo'] = None # Replace 'Nan' with None""")
        df_cleaned['sexo'] = df_cleaned['sexo'].astype(str).str.capitalize()
        # Handle possible 'nan' values converted to 'Nan'
        df_cleaned.loc[df_cleaned['sexo'] == 'Nan', 'sexo'] = None
        st.write("`sexo` values after normalization:")
        st.write(df_cleaned['sexo'].value_counts(dropna=False))
        st.markdown("**Justification:** Text is normalized to `Capitalize` to standardize 'Female'/'Male' and avoid capitalization or typographical inconsistencies. Null values converted to 'Nan' are reset to `None`.")

        st.markdown("#### Cleaning and Calculation of `fecha_nacimiento` and `edad`")
        st.code("""
# Convert 'fecha_nacimiento' to datetime, forcing nulls if the format is invalid
df_cleaned['fecha_nacimiento'] = pd.to_datetime(df_cleaned['fecha_nacimiento'], errors='coerce')

# Calculate 'edad' for nulls or inconsistent values
current_date = date.today() # Current date
df_cleaned['edad_calculada'] = df_cleaned['fecha_nacimiento'].apply(lambda dob: calculate_age_from_dob(dob, current_date))

# Prioritize calculated age if fecha_nacimiento is valid, otherwise use existing or None
df_cleaned['edad'] = df_cleaned.apply(
    lambda row: row['edad_calculada'] if pd.notna(row['edad_calculada']) else row['edad'], axis=1
)
df_cleaned['edad'] = df_cleaned['edad'].astype('Int64') # Int64 to allow NaNs and keep it integer
""")
        # Apply date and age cleaning
        df_cleaned['fecha_nacimiento'] = pd.to_datetime(df_cleaned['fecha_nacimiento'], errors='coerce')
        df_cleaned['edad_calculada'] = df_cleaned['fecha_nacimiento'].apply(lambda dob: calculate_age_from_dob(dob, current_date))
        df_cleaned['edad'] = df_cleaned.apply(
            lambda row: row['edad_calculada'] if pd.notna(row['edad_calculada']) else row['edad'], axis=1
        )
        df_cleaned['edad'] = df_cleaned['edad'].astype('Int64')
        df_cleaned = df_cleaned.drop(columns=['edad_calculada']) # Drop temporary column
        st.write("Null values in `edad` after cleaning:", df_cleaned['edad'].isna().sum())
        st.write("Null values in `fecha_nacimiento` after cleaning:", df_cleaned['fecha_nacimiento'].isna().sum())
        st.markdown("""
        **Justification:**
        - `fecha_nacimiento` is converted to `datetime` type, converting invalid formats to `NaT` (Not a Time).
        - `edad` is recalculated based on `fecha_nacimiento` if it's valid. This calculated age is prioritized if available. If `fecha_nacimiento` is `NaT`, the original `edad` is kept.
        - Ensures age is a non-negative integer. `Int64` is used to handle nulls in numeric columns.
        """)

        st.markdown("#### Cleaning `telefono`")
        st.code("""
# Remove non-numeric characters
df_cleaned['telefono'] = df_cleaned['telefono'].astype(str).str.replace(r'[^0-9]', '', regex=True)
# Replace empty strings (or just spaces) with None
df_cleaned.loc[df_cleaned['telefono'].str.strip() == '', 'telefono'] = None
""")
        df_cleaned['telefono'] = df_cleaned['telefono'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df_cleaned.loc[df_cleaned['telefono'].str.strip() == '', 'telefono'] = None # Replace empty strings with None
        st.write("Examples of `telefono` after cleaning:")
        st.dataframe(df_cleaned['telefono'].head())
        st.markdown("**Justification:** Non-numeric characters are removed from the phone to standardize the format. Empty strings or those with only spaces are converted to `None`.")

        st.subheader("2.2. Cross-Field Validations")
        st.markdown("Rules are applied to ensure logical consistency between columns.")

        st.markdown("#### Validation: `edad` consistent with `fecha_nacimiento`")
        # Recalculate age to compare with the final clean age
        df_cleaned_temp_age_check = df_cleaned.copy()
        df_cleaned_temp_age_check['calculated_age_for_check'] = df_cleaned_temp_age_check['fecha_nacimiento'].apply(lambda dob: calculate_age_from_dob(dob, current_date))

        inconsistent_ages = df_cleaned[
            (df_cleaned['edad'].notna()) &
            (df_cleaned_temp_age_check['calculated_age_for_check'].notna()) &
            (abs(df_cleaned['edad'] - df_cleaned_temp_age_check['calculated_age_for_check']) > 1) # Tolerance of 1 year for possible update discrepancies
        ]
        if not inconsistent_ages.empty:
            st.warning(f"**{len(inconsistent_ages)}** records were found with **inconsistent age** with the date of birth (difference > 1 year) *after cleaning*.")
            st.dataframe(inconsistent_ages[['id_paciente', 'fecha_nacimiento', 'edad']].head())
            st.markdown("""
            **Validation Rule:** The age calculated from `fecha_nacimiento` must be consistent with the reported `edad` (a small tolerance is allowed for possible date update discrepancies).
            **Action:** Cleaning already prioritizes the calculated age if `fecha_nacimiento` is valid, minimizing these inconsistencies. If they still exist, it could indicate an erroneous `fecha_nacimiento`.
            """)
        else:
            st.success("No significant inconsistencies were found between `edad` and `fecha_nacimiento` after cleaning.")

        st.markdown("#### Validation: `email` with valid format")
        invalid_email_after_clean = df_cleaned[~df_cleaned['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        if not invalid_email_after_clean.empty:
            st.warning(f"**{len(invalid_email_after_clean)}** records were found with **invalid email** after cleaning.")
            st.dataframe(invalid_email_after_clean[['id_paciente', 'email']].head())
            st.markdown("""
            **Validation Rule:** The `email` field must follow a standard email format (`text@text.domain`).
            **Action:** They are identified but not automatically modified, as this would require inference or manual interaction.
            """)
        else:
            st.success("All emails appear to have a valid format after cleaning (basic validation).")

        st.markdown("#### Validation: `telefono` contains only digits (after cleaning)")
        non_numeric_phones_cleaned = df_cleaned[df_cleaned['telefono'].notna() & ~df_cleaned['telefono'].astype(str).str.isdigit()]
        if not non_numeric_phones_cleaned.empty:
            st.warning(f"**{len(non_numeric_phones_cleaned)}** records were found with **phone numbers containing non-numeric characters** after cleaning (this should not happen if cleaning was effective).")
            st.dataframe(non_numeric_phones_cleaned[['id_paciente', 'telefono']].head())
        else:
            st.success("All phones contain only digits or are null after cleaning.")

        st.subheader("2.3. DataFrame After Cleaning")
        st.write("The first 10 rows of the clean DataFrame:")
        st.dataframe(df_cleaned.head(10))
        st.write("Information of the clean DataFrame:")
        buffer_cleaned = pd.io.common.StringIO()
        df_cleaned.info(buf=buffer_cleaned)
        s_cleaned = buffer_cleaned.getvalue()
        st.text(s_cleaned)

        # Save the clean and original DataFrame in the session state for use in the next section
        st.session_state['df_cleaned'] = df_cleaned
        st.session_state['df_original'] = df_pacientes

# Section 3: Quality Indicators and Documentation
elif selected_section == "3. Indicadores y Documentación":
    st.header("3. 📈 Indicadores de Calidad y Documentación")
    st.markdown("Summary of quality indicators before and after cleaning, along with documentation.")

    if 'df_cleaned' not in st.session_state or 'df_original' not in st.session_state:
        st.warning("Please navigate to the 'Limpieza y Validación' section first to generate clean data and session state.")
    else:
        df_original = st.session_state['df_original']
        df_cleaned = st.session_state['df_cleaned']

        st.subheader("3.1. Data Quality Indicators")

        # Function to calculate quality indicators
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

        st.markdown("#### Comparison of Missing Values (%)")
        cols = ['edad', 'fecha_nacimiento', 'telefono', 'sexo'] # Relevant columns for nulls
        data_nulos = {
            'Columna': cols,
            'Original (%)': [indicators_original['Valores Nulos por Columna (%)'].get(col, 0) for col in cols],
            'Limpio (%)': [indicators_cleaned['Valores Nulos por Columna (%)'].get(col, 0) for col in cols]
        }
        df_nulos_comp = pd.DataFrame(data_nulos)
        st.dataframe(df_nulos_comp.set_index('Columna'))

        st.markdown("""
        **Observations:**
        - A **significant reduction** in the percentage of nulls in `edad` is expected if `fecha_nacimiento` was available and valid.
        - `fecha_nacimiento` may show an increase in nulls if original formats were invalid and converted to `NaT`.
        - `telefono` may have nulls if empty strings remained after cleaning non-numeric characters.
        - `sexo` could have nulls if there were empty or non-standardizable values.
        """)

        st.markdown("#### Comparison of Data Types")
        st.write("Original data types:")
        st.json(indicators_original['Tipos de Datos por Columna'])
        st.write("Data types after cleaning:")
        st.json(indicators_cleaned['Tipos de Datos por Columna'])
        st.markdown("""
        **Observations:**
        - `fecha_nacimiento` should change from `object` (string) to `datetime64[ns]` (date and time type).
        - `edad` should change from `object` (if it contained nulls or was mixed) or `float64` (if numeric was inferred) to `Int64` (integer with null support).
        - `telefono` and `email` should ideally remain as `object` (string) but with validated format.
        """)

        st.markdown("#### Consistency and Uniqueness Indicators")
        st.write("**`sexo` - Category Uniqueness:**")
        st.write(f"Original: {df_original['sexo'].value_counts(dropna=False).index.tolist()}")
        st.write(f"Clean: {df_cleaned['sexo'].value_counts(dropna=False).index.tolist()}")
        st.markdown("""
        **Observation:** The number of unique categories and their names are expected to normalize after cleaning (e.g., only 'Female', 'Male' and `None`).
        """)

        st.write("**`email` - Format Pattern (Invalid Count):**")
        invalid_emails_original = df_original[~df_original['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        invalid_emails_cleaned = df_cleaned[~df_cleaned['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
        st.write(f"Invalid emails (Original): **{len(invalid_emails_original)}**")
        st.write(f"Invalid emails (Clean): **{len(invalid_emails_cleaned)}**")
        st.markdown("""
        **Observation:** Although cleaning does not alter them, their format was validated. This indicator shows whether non-standard format emails persist.
        """)

        st.subheader("3.2. Process Documentation")

        st.markdown("### **Assumptions Adopted During Cleaning:**")
        st.markdown("""
        * **Single Source for Age:** It's assumed that `fecha_nacimiento` is the most reliable source for determining `edad`. If `fecha_nacimiento` is valid, **calculating age from it is prioritized** over the existing `edad` value if it is null or inconsistent. Age is calculated as the difference in years to the current date, adjusting by month and day.
        * **`sexo` Format:** It's assumed that values `Female`, `female`, `Male`, `male`, and their variations should be standardized to `Female` and `Male` (capitalizing the first letter). Other values (`NaN`, empty, or unrecognized) are kept as `None`.
        * **`telefono` Format:** It's assumed that phone numbers should only contain digits. Any other characters (hyphens, spaces, parentheses, etc.) are **removed**. Empty strings or those consisting only of spaces resulting from this cleaning are interpreted as null (`None`).
        * **Date Coherence:** It's assumed that birth dates cannot be in the future or extremely old (age is calculated relative to the current date and negative ages are discarded, converting them to `None`).
        * **Patient ID:** It's assumed that `id_paciente` is the **unique** identifier for each patient and is not expected to have quality issues (duplicates, nulls).
        """)

        st.markdown("### **Validation Rules Implemented:**")
        st.markdown("""
        * **`fecha_nacimiento` Validation:** It's checked that the column can be converted to `datetime` type. Values that do not meet this format are marked as `NaT` (Not a Time).
        * **`edad` Validation:**
            * Must be a non-negative integer.
            * Must be **consistent** with `fecha_nacimiento`: the `edad` calculated from `fecha_nacimiento` must be close to the reported `edad` (a tolerance of 1 year is allowed for possible date update discrepancies in the original data).
        * **`sexo` Validation:** Values must be within a predefined set of standardized categories (`Female`, `Male`, or `None`).
        * **`email` Validation:** It's checked that the format follows a basic regular expression (`[^@]+@[^@]+\.[^@]+`) to ensure it contains an `@` and at least one `.` in the domain. This is a pattern validation, not an existence validation.
        * **`telefono` Validation:** It's checked that, after cleaning, the column only contains numeric characters (or is null).
        """)

        st.markdown("### **Improvement Recommendations to Ensure Future Data Quality:**")
        st.markdown("""
        1.  **Validation at Source:** Implement validations at the data entry level (e.g., web forms, databases) for `fecha_nacimiento`, `sexo`, `email`, and `telefono`.
            * **`fecha_nacimiento`:** Use date pickers to prevent erroneous manual entries and ensure `YYYY-MM-DD` format.
            * **`sexo`:** Use dropdowns with predefined options (`Female`, `Male`) to avoid capitalization inconsistencies or typos.
            * **`email`:** Implement real-time email format validation upon data entry and, if possible, a domain verification.
            * **`telefono`:** Enforce entry of digits only or a specific format (e.g., with input masks) depending on the country, and validate minimum/maximum length.
        2.  **`ciudad` Standardization:** Implement a catalog or master list of cities/municipalities to ensure consistency and avoid variations in city names (e.g., "Barranquilla" vs "barranquilla", or typos).
        3.  **Definition of Mandatory Fields:** Clearly establish which fields are mandatory (e.g., `id_paciente`, `nombre`, `fecha_nacimiento`, `sexo`) in the database or entry system to reduce the occurrence of critical null values.
        4.  **Regular Data Audits:** Conduct periodic audits of the database to identify new error patterns or data quality degradation over time.
        5.  **Metadata Documentation:** Maintain an updated `data dictionary` that clearly defines each field, its expected data type, format, validation rules, and meaning, accessible to the entire team.
        6.  **Error Reporting System:** Establish a mechanism for users (hospital staff, doctors) to report inconsistencies or data errors when they detect them, with a clear flow for correction.
        7.  **Staff Training:** Ensure that data entry personnel are continuously trained in best data entry practices and understand the importance of data quality for decision-making and patient care.
        """)

        st.subheader("3.3. Bonus (Optional)")
        st.markdown("""
        #### Implementation of Automatic Tests
        To implement automatic tests for data quality, frameworks like **Pytest** or **Great Expectations** could be used.

        **Conceptual example with Pytest (in a `tests/test_data_quality.py` file):**
        ```python
        # This code is conceptual and not part of app.py
        # You should have your cleaning and validation functions in a separate module to import them here.
        import pandas as pd
        import pytest
        from datetime import date
        # from your_project.data_quality_functions import clean_patient_data, calculate_age_from_dob # Example import

        # Ensure calculate_age_from_dob is accessible if you don't import it from a module
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
            # Test data with known cases to verify cleaning
            data = {
                "pacientes": [
                    {"id_paciente": 1, "nombre": "Claudia Torres", "fecha_nacimiento": "1954-01-08", "edad": None, "sexo": "Female", "email": "user1@example.com", "telefono": "342-950-1064", "ciudad": "Barranquilla"},
                    {"id_paciente": 2, "nombre": "Pedro Gomez", "fecha_nacimiento": "1980-05-15", "edad": 40, "sexo": "male", "email": "pedro@example", "telefono": "123-ABC-456", "ciudad": "Medellin"},
                    {"id_paciente": 3, "nombre": "Ana Smith", "fecha_nacimiento": "2025-01-01", "edad": 5, "sexo": "FEMALE", "email": "ana@example.com", "telefono": "7891234567", "ciudad": "Bogota"}, # Future date, incorrect age
                    {"id_paciente": 4, "nombre": "Luis Lopez", "fecha_nacimiento": "1990-11-20", "edad": None, "sexo": "Male", "email": "luis.lopez@example.com", "telefono": "9876543210", "ciudad": "Cali"},
                    {"id_paciente": 5, "nombre": "Maria Paz", "fecha_nacimiento": None, "edad": 30, "sexo": "Female", "email": "maria@example.net", "telefono": "300-111-2222", "ciudad": "Bogota"},
                    {"id_paciente": 6, "nombre": "Carlos", "fecha_nacimiento": "1970-07-08", "edad": 50, "sexo": "OTHER", "email": "carlos@example.com", "telefono": "123-456-7890", "ciudad": "Cartagena"}
                ]
            }
            return pd.json_normalize(data['pacientes'])

        # This would be the cleaning function you would test, adapted from your app.py
        def clean_patient_data_for_test(df_raw):
            df_cleaned_test = df_raw.copy()
            current_date_for_test = date(2025, 7, 8) # Fixed date for age tests

            # Sexo
            df_cleaned_test['sexo'] = df_cleaned_test['sexo'].astype(str).str.capitalize()
            df_cleaned_test.loc[df_cleaned_test['sexo'] == 'Nan', 'sexo'] = None
            # Also clean non-standard values like 'Other'
            df_cleaned_test.loc[~df_cleaned_test['sexo'].isin(['Female', 'Male', None]), 'sexo'] = None

            # Fecha Nacimiento and Edad
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
            assert all(s in ['Female', 'Male', None] for s in df_cleaned['sexo'].unique()), "Sex values are not standardized or contain unexpected ones."
            assert pd.isna(df_cleaned.loc[df_cleaned['id_paciente'] == 6, 'sexo'].iloc[0]), "The 'OTHER' value for sex was not converted to None."


        def test_age_calculation_and_validation(sample_patient_data):
            df_cleaned = clean_patient_data_for_test(sample_patient_data)
            assert all(df_cleaned['edad'].dropna() >= 0), "Calculated ages must not be negative."
            # Verify age for id_paciente 1 (1954-01-08) -> 2025-1954 = 71
            assert df_cleaned.loc[df_cleaned['id_paciente'] == 1, 'edad'].iloc[0] == 71, "Age for patient 1 was not calculated correctly."
            # Verify that future date (2025-01-01) results in null/None age
            assert pd.isna(df_cleaned.loc[df_cleaned['id_paciente'] == 3, 'edad'].iloc[0]), "Age for future date should be null."
            # Verify that if fecha_nacimiento is null but age exists, it is kept (id_paciente 5)
            assert df_cleaned.loc[df_cleaned['id_paciente'] == 5, 'edad'].iloc[0] == 30, "Age for patient 5 was not kept correctly."


        def test_email_format_after_cleaning(sample_patient_data):
            df_cleaned = clean_patient_data_for_test(sample_patient_data) # No direct email cleaning, just validation
            invalid_emails_in_cleaned = df_cleaned[~df_cleaned['email'].astype(str).str.match(r'[^@]+@[^@]+\.[^@]+', na=False)]
            # We expect 'pedro@example' to still be invalid
            assert 'pedro@example' in invalid_emails_in_cleaned['email'].values, "Email 'pedro@example' was not marked as invalid."
            # We don't expect new invalid ones to be added, and their number should be consistent with the originals
            assert len(invalid_emails_in_cleaned) == 1, "An unexpected number of invalid emails were detected."


        def test_telefono_numeric_after_cleaning(sample_patient_data):
            df_cleaned = clean_patient_data_for_test(sample_patient_data)
            assert all(df_cleaned['telefono'].dropna().apply(lambda x: x.isdigit())), "The phone field contains non-numeric characters after cleaning."
            assert pd.isna(df_cleaned.loc[df_cleaned['id_paciente'] == 2, 'telefono'].iloc[0]), "Phone with non-numeric characters was not cleaned correctly."
        ```
        To run Pytest, you need:
        1.  `pytest` installed: `pip install pytest`
        2.  Save the test code in a file like `tests/test_data_quality.py` (or similar) in a `tests/` folder.
        3.  **Important:** Refactor your cleaning and validation functions from `app.py` into a separate Python module (e.g., `data_processing.py`) so you can import them in the tests. Or, for this demonstration, you can copy and adapt the cleaning functions within the test file itself as shown above.
        4.  Run `pytest` in your terminal from the root of your project.

        #### Simulation of Clean Data Migration to a Target Structure
        Once the data has been cleaned and validated, the next logical step in a data pipeline is to load it into a target structure, such as a Data Warehouse or an analytical database. Formats like **Parquet** are ideal for this due to their columnar nature, efficient compression, and ability to handle complex schemas.

        Here we simulate downloading the clean data in common formats for migration.
        """)

        # Code for the download bonus
        if 'df_cleaned' in st.session_state:
            st.markdown("#### Simulation of Clean Data Migration")

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            @st.cache_data
            def convert_df_to_parquet(df):
                try:
                    import pyarrow # Ensure pyarrow is installed for to_parquet
                    return df.to_parquet(index=False)
                except ImportError:
                    st.error("To download in Parquet format, install 'pyarrow': `pip install pyarrow`")
                    return None
                except Exception as e:
                    st.error(f"Error generating Parquet file: {e}")
                    return None

            col_csv, col_parquet = st.columns(2)
            with col_csv:
                st.download_button(
                    label="Download Clean Data (CSV)",
                    data=convert_df_to_csv(st.session_state['df_cleaned']),
                    file_name="pacientes_limpios.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            with col_parquet:
                parquet_data = convert_df_to_parquet(st.session_state['df_cleaned'])
                if parquet_data is not None: # Only show the button if parquet data was successfully generated
                    st.download_button(
                        label="Download Clean Data (Parquet)",
                        data=parquet_data,
                        file_name="pacientes_limpios.parquet",
                        mime="application/octet-stream",
                        key="download_parquet"
                    )

            st.markdown("""
            **Migration Justification:**
            Migrating clean data to a Data Warehouse (DW) typically involves:
            1.  **Extract:** Obtain data from sources.
            2.  **Transform:** Data is cleaned, standardized, validated, and prepared to fit the DW schema. This is the phase we've detailed in this application.
            3.  **Load:** Transformed data is loaded into the DW's dimensional and fact tables.
            Formats like CSV are universal, but Parquet is preferred in Big Data and DW environments for its efficiency. The CSV/Parquet download simulation represents the output of this transformation process, ready to be loaded into a system optimized for analytical queries.
            """)

# --- New Section 4: Advanced EDA and Dashboards ---
elif selected_section == "4. EDA Avanzado & Dashboards":
    st.header("4. 📊 Advanced EDA and Interactive Dashboards")
    st.markdown("Deep exploration of clean data and creation of interactive visualizations.")

    if 'df_cleaned' not in st.session_state:
        st.warning("Please navigate to the 'Limpieza y Validación' section first to load the clean data.")
    else:
        df_display = st.session_state['df_cleaned'].copy()

        st.subheader("Dashboard Filters")
        col1, col2, col3 = st.columns(3)

        # Filter by city
        all_cities = ['Todas'] + sorted(df_display['ciudad'].dropna().unique().tolist())
        selected_city_filter = col1.selectbox("Filter by City:", all_cities)
        if selected_city_filter != 'Todas':
            df_display = df_display[df_display['ciudad'] == selected_city_filter]

        # Filter by gender
        all_sex = ['Todos'] + sorted(df_display['sexo'].dropna().unique().tolist())
        selected_sex_filter = col2.selectbox("Filter by Sex:", all_sex)
        if selected_sex_filter != 'Todos':
            df_display = df_display[df_display['sexo'] == selected_sex_filter]

        # Filter by age range
        # Ensure the filtered df_display has values for min/max
        if not df_display['edad'].dropna().empty:
            min_age_data = int(df_display['edad'].min())
            max_age_data = int(df_display['edad'].max())
            age_range = col3.slider("Age Range:", min_value=min_age_data, max_value=max_age_data, value=(min_age_data, max_age_data))
            df_display = df_display[(df_display['edad'] >= age_range[0]) & (df_display['edad'] <= age_range[1])]
        else:
            col3.info("No ages available for filtering.")


        st.subheader("Key Metrics (KPIs)")
        num_patients = len(df_display)
        avg_age = df_display['edad'].mean()
        most_common_city = df_display['ciudad'].mode()[0] if not df_display['ciudad'].empty else "N/A"

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Total Patients (Filtered)", num_patients)
        with kpi2:
            st.metric("Average Age (Filtered)", f"{avg_age:.1f}" if not pd.isna(avg_age) else "N/A")
        with kpi3:
            st.metric("Most Common City", most_common_city)

        st.subheader("Detailed Visualizations")

        if not df_display.empty:
            # Age Distribution (Box Plot and Violin Plot)
            st.markdown("#### Age Distribution")
            fig_age, axes_age = plt.subplots(1, 2, figsize=(16, 6))
            sns.boxplot(y=df_display['edad'].dropna(), ax=axes_age[0], color="skyblue")
            axes_age[0].set_title('Age Box Plot')
            axes_age[0].set_ylabel('Age')
            sns.violinplot(y=df_display['edad'].dropna(), ax=axes_age[1], color="lightgreen")
            axes_age[1].set_title('Age Violin Plot')
            axes_age[1].set_ylabel('Age')
            st.pyplot(fig_age)
            st.markdown("Box plots and violin plots help visualize age distribution and the presence of outliers.")

            # Gender Distribution by City
            st.markdown("#### Patient Distribution by Gender and City")
            # Ensure there's data to group
            if not df_display[['ciudad', 'sexo']].dropna().empty:
                sex_city_counts = df_display.groupby(['ciudad', 'sexo']).size().unstack(fill_value=0)
                fig_sex_city, ax_sex_city = plt.subplots(figsize=(12, 7))
                sex_city_counts.plot(kind='bar', stacked=True, ax=ax_sex_city, cmap='Pastel1')
                ax_sex_city.set_title('Number of Patients by City and Gender')
                ax_sex_city.set_xlabel('City')
                ax_sex_city.set_ylabel('Number of Patients')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig_sex_city)
                st.dataframe(sex_city_counts)
                st.markdown("This stacked bar chart shows the gender composition within each city.")
            else:
                st.info("No sufficient data to generate the Gender by City chart with current filters.")

            # Average Age by City and Gender
            st.markdown("#### Average Age by City and Gender")
            if not df_display[['ciudad', 'sexo', 'edad']].dropna().empty:
                avg_age_city_sex = df_display.groupby(['ciudad', 'sexo'])['edad'].mean().unstack()
                fig_avg_age, ax_avg_age = plt.subplots(figsize=(12, 7))
                sns.heatmap(avg_age_city_sex, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, ax=ax_avg_age)
                ax_avg_age.set_title('Average Age by City and Gender')
                st.pyplot(fig_avg_age)
                st.dataframe(avg_age_city_sex)
                st.markdown("A heatmap to quickly visualize the average age in different city and gender combinations.")
            else:
                st.info("No sufficient data to generate the Average Age by City and Gender heatmap with current filters.")

        else:
            st.info("No data to display with the selected filters.")

# --- New Section 5: Machine Learning Modeling (Clustering) ---
elif selected_section == "5. Modelado de Machine Learning":
    st.header("5. 🧠 Machine Learning Modeling: Patient Clustering (Clustering)")
    st.markdown("Identification of patient segments with similar characteristics using K-Means.")

    if 'df_cleaned' not in st.session_state:
        st.warning("Please navigate to the 'Limpieza y Validación' section first to load the clean data.")
    else:
        df_ml = st.session_state['df_cleaned'].copy()

        st.subheader("Data Preparation for ML")
        # Select numeric features for clustering
        features = ['edad'] # For now, only age. If you have more, add them here.

        # Remove rows with nulls in the selected features (for clustering)
        # Convert 'edad' to float to handle NaN if any remain (KMeans doesn't directly accept them)
        df_ml['edad'] = df_ml['edad'].astype(float)
        df_ml_filtered = df_ml.dropna(subset=features)

        if df_ml_filtered.empty:
            st.warning("Not enough clean and complete data to perform clustering with the selected features.")
        else:
            X = df_ml_filtered[features]

            # Feature Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            st.write("Scaled data for the clustering model (first 5 rows):")
            st.dataframe(pd.DataFrame(X_scaled, columns=features).head())
            st.markdown("**Justification:** Scaling is crucial for distance-based algorithms like K-Means, ensuring no feature dominates due to its scale.")

            st.subheader("Determining the Optimal Number of Clusters (Elbow Method)")

            # --- Manual Elbow Method Implementation ---
            sse = [] # Sum of Squared Errors (or Inertia)
            # Test a range of K from 1 to 10 (or adjust as needed)
            k_range = range(1, 11)

            for k in k_range:
                try:
                    # n_init='auto' is the recommended value for modern KMeans
                    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    kmeans_model.fit(X_scaled)
                    sse.append(kmeans_model.inertia_)
                except ValueError as e:
                    # Catch error if k=1 and there's only one feature (e.g., 'edad')
                    # This can happen with KMeans for n_clusters=1 on some sklearn versions
                    if k == 1:
                        sse.append(0) # Inertia is 0 for 1 cluster if all points are the center
                    else:
                        st.error(f"Error calculating inertia for k={k}: {e}")
                        sse.append(None) # Append None if an error occurs for other k

            # Plotting the Elbow Method
            fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
            ax_elbow.plot(k_range, sse, marker='o', linestyle='--')
            ax_elbow.set_title('Elbow Method for K-Means')
            ax_elbow.set_xlabel('Number of Clusters (k)')
            ax_elbow.set_ylabel('Inertia (SSE)')
            ax_elbow.grid(True)
            st.pyplot(fig_elbow)
            st.markdown("""
            The **Elbow Method** helps determine the optimal number of clusters (`k`). You look for the point on the graph where the inertia (sum of squared distances within the cluster) significantly decreases, forming an "elbow" or "knee".
            """)
            # --- End of Manual Elbow Method Implementation ---

            # Slider for the user to choose the number of clusters
            st.subheader("K-Means Model Configuration")
            n_clusters = st.slider("Select the number of clusters (k):", min_value=2, max_value=8, value=3)

            # Train the model
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            df_ml_filtered['cluster'] = kmeans.fit_predict(X_scaled)
            st.success(f"K-Means model trained with **{n_clusters}** clusters.")

            st.subheader("Clustering Results")

            # Average features per cluster
            cluster_centers_scaled = kmeans.cluster_centers_
            cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled) # Convert back to original scale
            cluster_df = pd.DataFrame(cluster_centers_original, columns=features)
            cluster_df['Cluster'] = range(n_clusters)
            st.markdown("#### Average Features per Cluster (in original scale)")
            st.dataframe(cluster_df.set_index('Cluster'))
            st.markdown("These values represent the center of each cluster, helping to interpret what defines each patient group.")

            # Patient count per cluster
            st.markdown("#### Patient Count per Cluster")
            cluster_counts = df_ml_filtered['cluster'].value_counts().sort_index()
            fig_cluster_counts, ax_cluster_counts = plt.subplots(figsize=(8, 5))
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax_cluster_counts, palette="viridis")
            ax_cluster_counts.set_title('Number of Patients per Cluster')
            ax_cluster_counts.set_xlabel('Cluster')
            ax_cluster_counts.set_ylabel('Patient Count')
            st.pyplot(fig_cluster_counts)
            st.dataframe(cluster_counts.to_frame(name='Count'))
            st.markdown("This chart shows how many patients were assigned to each cluster.")

            # Visualization of clusters (if we only have one feature like 'edad')
            st.markdown("#### Cluster Visualization (Age Distribution per Cluster)")
            fig_cluster_dist, ax_cluster_dist = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df_ml_filtered, x='edad', hue='cluster', kde=True, palette='tab10', ax=ax_cluster_dist, bins=15)
            ax_cluster_dist.set_title('Age Distribution per Cluster')
            ax_cluster_dist.set_xlabel('Age')
            ax_cluster_dist.set_ylabel('Frequency')
            st.pyplot(fig_cluster_dist)
            st.markdown("This superimposed histogram shows how ages are distributed within each cluster, helping to understand the age profiles of each group.")

            st.markdown("### **Interpretation and Applications:**")
            st.markdown(f"""
            Based on `edad`, the K-Means model has identified **{n_clusters}** distinct patient groups.
            For example, if the clusters are:
            * **Cluster 0:** Could represent young patients (e.g., average age 20-30 years).
            * **Cluster 1:** Could represent middle-aged patients (e.g., average age 40-50 years).
            * **Cluster 2:** Could represent older patients (e.g., average age 60+ years).

            **Potential applications:**
            * **Personalized Marketing and Communication:** Send relevant information about prevention or specific health programs for each age group.
            * **Hospital Resource Management:** Anticipate the needs of certain age groups (e.g., pediatric specialties for the young cluster, geriatrics for the older cluster).
            * **Clinical Research:** Study disease patterns or treatments that are more prevalent in a particular age segment.
            """)
