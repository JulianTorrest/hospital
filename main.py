import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# URL del archivo JSON
DATA_URL = "https://raw.githubusercontent.com/JulianTorrest/hospital/refs/heads/main/dataset_hospital%202.json"

st.set_page_config(layout="wide") # Configurar el layout de la página para usar todo el ancho

st.title("🏥 Análisis Exploratorio de Datos (EDA) de Pacientes de Hospital")
st.markdown("Esta aplicación realiza un análisis exploratorio de datos sobre un conjunto de datos de pacientes de un hospital.")

@st.cache_data # Decorador para cachear la función y mejorar el rendimiento
def load_data(url):
    """
    Carga los datos desde la URL proporcionada.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza una excepción si la solicitud falla
        data = response.json()
        df = pd.json_normalize(data['pacientes']) # Normalizar el JSON para obtener un DataFrame
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error al cargar los datos desde la URL: {e}")
        return pd.DataFrame() # Retorna un DataFrame vacío en caso de error
    except KeyError:
        st.error("El archivo JSON no contiene la clave 'pacientes'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al procesar los datos: {e}")
        return pd.DataFrame()

df = load_data(DATA_URL)

if not df.empty:
    st.header("📋 Vista Previa de los Datos")
    st.write("Las primeras 5 filas del conjunto de datos:")
    st.dataframe(df.head())

    st.header("📊 Información General del Dataset")
    buffer = pd.io.common.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Estadísticas Descriptivas")
    st.dataframe(df.describe(include='all'))

    # --- Preparación de Datos ---
    st.header("⚙️ Preparación de Datos")
    st.markdown("Realizando la conversión de `fecha_nacimiento` y calculando `edad` si es nulo.")

    # Convertir 'fecha_nacimiento' a datetime
    df['fecha_nacimiento'] = pd.to_datetime(df['fecha_nacimiento'], errors='coerce')

    # Calcular 'edad' si es nulo o si 'fecha_nacimiento' es válida
    current_year = datetime.now().year
    df['edad'] = df.apply(lambda row: current_year - row['fecha_nacimiento'].year if pd.isna(row['edad']) and pd.notna(row['fecha_nacimiento']) else row['edad'], axis=1)
    df['edad'] = df['edad'].astype('Int64') # Usar Int64 para permitir NaNs

    st.write("Columnas después de la preparación:")
    st.dataframe(df[['fecha_nacimiento', 'edad']].head())
    st.write(f"Número de valores nulos en 'edad' después del procesamiento: {df['edad'].isna().sum()}")


    # --- Análisis Exploratorio de Datos ---
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")

    # Distribución de 'sexo'
    st.subheader("Distribución por Sexo")
    sex_counts = df['sexo'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax1.axis('equal') # Para que el círculo sea un círculo.
    st.pyplot(fig1)
    st.write(sex_counts)

    # Distribución de 'ciudad'
    st.subheader("Distribución por Ciudad")
    city_counts = df['ciudad'].value_counts().head(10) # Mostrar las top 10 ciudades
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=city_counts.index, y=city_counts.values, ax=ax2, palette="viridis")
    ax2.set_title('Top 10 Ciudades de Residencia')
    ax2.set_xlabel('Ciudad')
    ax2.set_ylabel('Número de Pacientes')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig2)
    st.write(city_counts)

    # Distribución de 'edad'
    st.subheader("Distribución de Edad")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['edad'].dropna(), kde=True, ax=ax3, bins=20, color="skyblue")
    ax3.set_title('Distribución de Edades de los Pacientes')
    ax3.set_xlabel('Edad')
    ax3.set_ylabel('Frecuencia')
    st.pyplot(fig3)
    st.write(df['edad'].describe())

    # Agrupar por sexo y ver la edad media
    st.subheader("Edad Media por Sexo")
    avg_age_by_sex = df.groupby('sexo')['edad'].mean().reset_index()
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.barplot(x='sexo', y='edad', data=avg_age_by_sex, ax=ax4, palette="coolwarm")
    ax4.set_title('Edad Media por Sexo')
    ax4.set_xlabel('Sexo')
    ax4.set_ylabel('Edad Media')
    st.pyplot(fig4)
    st.dataframe(avg_age_by_sex)

    # Interacción: Filtrar por ciudad
    st.header("🗺️ Exploración por Ciudad")
    selected_city = st.selectbox(
        "Selecciona una ciudad para ver el número de pacientes:",
        options=['Todas'] + sorted(df['ciudad'].dropna().unique().tolist())
    )

    if selected_city == 'Todas':
        st.write(f"Número total de pacientes: {len(df)}")
    else:
        patients_in_city = df[df['ciudad'] == selected_city]
        st.write(f"Número de pacientes en **{selected_city}**: {len(patients_in_city)}")
        st.dataframe(patients_in_city.head())

else:
    st.error("No se pudieron cargar los datos o el DataFrame está vacío.")
