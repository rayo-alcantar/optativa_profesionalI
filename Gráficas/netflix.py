# Ángel De Jesús Alcantar Garza
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Cargar el dataset desde GitHub
url = "https://raw.githubusercontent.com/rayo-alcantar/optativa_profesionalI/main/datos/Netflix.csv"
df = pd.read_csv(url)

# 2. Mostrar la estructura del DataFrame
print(df.dtypes)

# 3. Limpiar y convertir la columna 'date_added' a formato datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
print("Columna 'date_added' convertida a datetime correctamente.")
print(df['date_added'].dtype)
print(df['date_added'].head())

# Mapeo de rating a valores numéricos
rating_map = {
    'G': 1,
    'PG': 2,
    'PG-13': 3,
    'R': 4,
    'NC-17': 5,
    'TV-Y': 1,
    'TV-Y7': 2,
    'TV-G': 1,
    'TV-PG': 2,
    'TV-14': 3,
    'TV-MA': 4
}

# Crear una nueva columna 'rating_numeric' usando el mapeo de 'rating_map'
df['rating_numeric'] = df['rating'].map(rating_map).fillna(0)

# Verificar las primeras filas para confirmar la creación de 'rating_numeric'
print(df[['rating', 'rating_numeric']].head())

# 4. Manejo de datos faltantes y duplicados
# Eliminar filas donde falte la columna 'director'
df.dropna(subset=['director'], inplace=True)

# Eliminar filas duplicadas
df.drop_duplicates(inplace=True)

# 5. Estadísticas descriptivas de las columnas numéricas
numeric_columns = ['release_year', 'duration', 'rating_numeric']
print("\nEstadísticas descriptivas para variables numéricas:")
print(df[numeric_columns].describe())

# Medidas de tendencia central
print("\nMedidas de tendencia central para variables numéricas:")
for column in numeric_columns:
    print(f"\n{column}:")
    print(f"Media: {df[column].mean():.2f}")
    print(f"Mediana: {df[column].median():.2f}")
    print(f"Moda: {df[column].mode().values[0]:.2f}")

# Variables categóricas a analizar
categorical_columns = ['type', 'rating', 'genres']
print("\nEstadísticas para variables categóricas:")
for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts())
    print(f"Moda: {df[column].mode().values[0]}")

# Estadísticas para la columna de fecha
print("\nEstadísticas para la columna de fecha:")
print(f"Fecha más temprana: {df['date_added'].min()}")
print(f"Fecha más reciente: {df['date_added'].max()}")
print(f"Rango de fechas: {df['date_added'].max() - df['date_added'].min()}")

# 6. Conteo de valores únicos para columnas de tipo objeto
object_columns = ['show_id', 'title', 'director', 'cast', 'country', 'description']
print("\nConteo de valores únicos para otras columnas de objeto:")
for column in object_columns:
    print(f"{column}: {df[column].nunique()} valores únicos")

# 7. Gráfico de barras para la columna 'rating'
rating_counts = df['rating'].value_counts()

plt.figure(figsize=(10, 6))
rating_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Frecuencia de Ratings en el Dataset de Netflix', fontsize=16)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.tight_layout()
plt.show()

# Interpretación del gráfico
# Podemos ver que el rating más común en el dataset es "TV-MA" con más de 1750 títulos,
# lo que indica que Netflix ofrece una gran cantidad de contenido dirigido a audiencias adultas.
# El segundo rating más frecuente es "TV-14", con aproximadamente 1250 títulos,
# lo cual sugiere que también existe una considerable oferta de contenido adecuado para adolescentes.
# Los ratings enfocados en audiencias más jóvenes, como "TV-G", "TV-Y" y "G", tienen menor presencia.

# 8. Gráfico de pastel para la columna 'type'
plt.figure(figsize=(16, 10))
sns.set_style("whitegrid")

plt.subplot(2, 3, 4)
df['type'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen'], explode=(0.05, 0))
plt.title('Proporción de Películas vs Series', fontsize=16)

plt.tight_layout()
plt.show()

# Interpretación del gráfico de pastel
# Podemos observar que la gran mayoría del contenido en Netflix, 96.6%, corresponde a películas,
# mientras que solo un 3.4% son series. Esto indica que Netflix se enfoca en ofrecer películas,
# aunque las series también están presentes en menor cantidad.

# Gráfico de barras para mostrar el Top 10 de países con más producciones
plt.figure(figsize=(16, 10))
sns.set_style("whitegrid")

plt.subplot(2, 3, 5)
df['country'].value_counts().head(10).plot(kind='bar', color='skyblue', edgecolor='black')

# Ajustes del gráfico
plt.title('Top 10 Países con Más Producciones', fontsize=16)
plt.xlabel('País', fontsize=12)
plt.ylabel('Cantidad', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Ajustar el diseño y mostrar el gráfico
plt.tight_layout()
plt.show()
#En el gráfico podemos observar que Estados Unidos es, por un amplio margen, el país con más producciones en el catálogo de Netflix, con más de 1750 títulos. Le sigue India con alrededor de 850 producciones, mientras que el resto de los países, como Reino Unido, Canadá y Francia, tienen una cantidad considerablemente menor, con menos de 400 producciones cada uno. Esto refleja una fuerte concentración de contenido producido en los Estados Unidos, lo cual sugiere una predominancia de producciones de habla inglesa y contenido proveniente de Hollywood y la industria del entretenimiento estadounidense.


# Gráfico de barras para mostrar el Top 10 de géneros más comunes
plt.figure(figsize=(16, 10))
sns.set_style("whitegrid")

# Contar la frecuencia de los géneros más comunes y seleccionar los 10 principales
genre_counts = df['genres'].value_counts().head(10)

# Crear el gráfico de barras
genre_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Ajustes del gráfico
plt.title('Top 10 Géneros Más Comunes en Netflix', fontsize=16)
plt.xlabel('Género', fontsize=12)
plt.ylabel('Cantidad', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Ajustar el diseño y mostrar el gráfico
plt.tight_layout()
plt.show()
