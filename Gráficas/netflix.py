# Ángel De Jesús Alcántar Garza
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
df.dropna(subset=['director'], inplace=True)
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
#En este gráfico podemos observar que el rating más común es "TV-MA", lo cual indica que gran parte del contenido en Netflix está destinado a audiencias maduras. Esto es consistente con la tendencia actual de plataformas de streaming que ofrecen una amplia variedad de contenido orientado a adultos. En comparación, otros ratings como "G" y "NC-17" tienen una frecuencia mucho menor, lo que sugiere que el contenido para niños o con restricciones extremas es más limitado en la plataforma.

# 8. Gráfico de pastel para películas y series
plt.figure(figsize=(16, 10))
sns.set_style("whitegrid")

plt.subplot(2, 3, 4)
df['type'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen'], explode=(0.05, 0))
plt.title('Proporción de Películas vs Series', fontsize=16)

plt.tight_layout()
plt.show()
Comentario sobre el gráfico de pastel de 'Proporción de Películas vs Series':
En este gráfico podemos observar que la mayor parte del contenido en Netflix está compuesto por películas, representando aproximadamente el 60% del total, mientras que las series constituyen el 40%. Esto sugiere que aunque Netflix es bien conocido por sus series originales, todavía tiene una mayor oferta de películas en su catálogo. Esta información puede ser útil para aquellos interesados en la oferta de contenido y la diversidad entre películas y series dentro de la plataforma.

# Gráfico de barras para mostrar el Top 10 de países con más producciones
plt.figure(figsize=(16, 10))
sns.set_style("whitegrid")

plt.subplot(2, 3, 5)
df['country'].value_counts().head(10).plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Top 10 Países con Más Producciones', fontsize=16)
plt.xlabel('País', fontsize=12)
plt.ylabel('Cantidad', fontsize=12)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
#En este gráfico podemos observar que Estados Unidos lidera con diferencia en la cantidad de producciones disponibles en Netflix, seguido de otros países como India y Reino Unido. Esto refleja la fuerte presencia de la industria cinematográfica y televisiva estadounidense en el catálogo de Netflix, mientras que otros países como India y Reino Unido también tienen una participación significativa. Este dominio de Estados Unidos es coherente con su posición en la industria del entretenimiento a nivel global.


# Gráfico de barras para mostrar el Top 10 de géneros más comunes
plt.figure(figsize=(16, 10))
sns.set_style("whitegrid")

# Contar la frecuencia de los géneros más comunes y seleccionar los 10 principales
genre_counts = df['genres'].value_counts().head(10)

# Crear el gráfico de barras
genre_counts.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Top 10 Géneros Más Comunes en Netflix', fontsize=16)
plt.xlabel('Género', fontsize=12)
plt.ylabel('Cantidad', fontsize=12)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
#En este gráfico podemos observar que los géneros más comunes en Netflix incluyen las comedias, los dramas y las películas de acción. Esto sugiere que la plataforma ofrece una amplia variedad de contenido en estos géneros, probablemente debido a su popularidad entre los usuarios. También vemos una diversidad de géneros que abarca tanto producciones internacionales como contenido orientado a públicos específicos, lo que indica la amplitud del catálogo de Netflix.

# 9. Duración promedio por tipo de contenido
# Asegurarse de que la columna 'duration' esté en formato numérico
df['duration_numeric'] = df['duration'].apply(lambda x: int(str(x).split(' ')[0]) if pd.notnull(x) else 0)

# Calcular la duración promedio por tipo de contenido ('type')
avg_duration_by_type = df.groupby('type')['duration_numeric'].mean().reset_index()

# Visualizar el gráfico de barras para la duración promedio por tipo de contenido
plt.figure(figsize=(10, 6))
sns.barplot(x='type', y='duration_numeric', data=avg_duration_by_type, palette='Blues')

plt.title('Duración Promedio por Tipo de Contenido en Netflix', fontsize=16)
plt.xlabel('Tipo de Contenido', fontsize=12)
plt.ylabel('Duración Promedio (minutos)', fontsize=12)
plt.tight_layout()
plt.show()
#En este gráfico podemos observar que las películas tienen una duración promedio significativamente mayor que las series, lo cual es lógico, dado que las películas son producciones autoconclusivas, mientras que las series se componen de episodios más cortos. La diferencia en la duración promedio puede influir en la experiencia del espectador dependiendo de si prefiere contenido más largo (películas) o más corto y dividido en episodios (series).

# Gráfico de dispersión para la duración versus el año de lanzamiento
plt.figure(figsize=(16, 10))
sns.scatterplot(data=df, x='release_year', y='duration', hue='type', size='rating_numeric', sizes=(20, 200), alpha=0.7)
plt.title('Duración vs Año de Lanzamiento', fontsize=20)
plt.xlabel('Año de Lanzamiento', fontsize=16)
plt.ylabel('Duración (minutos)', fontsize=16)
plt.legend(title='Tipo', title_fontsize='13', fontsize='11')
plt.tight_layout()
plt.show()

# Seleccionar solo las columnas de interés para el heatmap
heatmap_data = df[['release_year', 'duration_numeric', 'rating_numeric']]

# Calcular la matriz de correlación
correlation_matrix = heatmap_data.corr()

# Crear el heatmap usando Seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)

plt.title('Mapa de Calor de Correlaciones entre Año de Lanzamiento, Duración y Rating', fontsize=16)
plt.tight_layout()

# Mostrar el gráfico
plt.show()
#En este mapa de calor podemos observar que no existe una correlación fuerte entre las variables analizadas. Las correlaciones entre el año de lanzamiento, la duración, y el rating son relativamente bajas, lo que indica que estas variables no están fuertemente relacionadas entre sí en este dataset. Esto sugiere que la duración de un contenido o su calificación no depende significativamente del año de lanzamiento.



# Agrupar los datos por 'release_year' y 'rating_numeric' y calcular el promedio de rating por año
ratings_by_year = df.groupby('release_year')['rating_numeric'].mean().reset_index()

# Crear el gráfico de líneas para mostrar la evolución de los ratings a lo largo de los años
plt.figure(figsize=(10, 6))
sns.lineplot(x='release_year', y='rating_numeric', data=ratings_by_year, marker='o', color='skyblue')

# Ajustes del gráfico
plt.title('Evolución de los Ratings a lo Largo de los Años', fontsize=16)
plt.xlabel('Año de Lanzamiento', fontsize=12)
plt.ylabel('Rating Promedio', fontsize=12)
plt.grid(True)
plt.tight_layout()

# Mostrar el gráfico
plt.show()
#En este gráfico vemos que la mayoría de las películas y series en Netflix se concentran en los años más recientes, particularmente a partir del 2015, lo que refleja la tendencia de las plataformas de streaming de invertir más en nuevas producciones. Además, las películas suelen tener una mayor variabilidad en la duración, mientras que las series se agrupan en duraciones más cortas. Este patrón es consistente con la estructura típica de una serie de episodios.

# Contar el número de títulos por género y año de lanzamiento
genres_by_year = df.groupby(['release_year', 'genres']).size().reset_index(name='count')

# Crear el gráfico de líneas para mostrar la tendencia de los géneros a lo largo de los años
plt.figure(figsize=(16, 10))
sns.lineplot(x='release_year', y='count', hue='genres', data=genres_by_year, marker='o')

# Ajustes del gráfico
plt.title('Tendencia de Géneros a lo Largo de los Años', fontsize=16)
plt.xlabel('Año de Lanzamiento', fontsize=12)
plt.ylabel('Número de Títulos', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar el gráfico
plt.show()
#En este gráfico observamos que los ratings promedio de los contenidos en Netflix se han mantenido relativamente constantes a lo largo de los años, con algunas variaciones menores. No se observa una tendencia clara hacia un aumento o disminución en las calificaciones. Esto sugiere que el tipo de contenido en términos de clasificación ha permanecido estable, independientemente del año de lanzamiento.
