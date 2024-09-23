#importamos pandas
#angel alcantar>rayoalcantar@gmail.com.

import pandas as pd

data = {
	'Nombre': ['Ana', 'Juan', 'María', 'Carlos', 'Sofía'],
	'Edad': [25, 30, 22, 28, 35],
	'Ciudad': ['Madrid', 'Barcelona', 'Madrid', 'Valencia', 'Barcelona'],
	'Puntuación': [85, 92, 78, 95, 88]
}

# Crear DataFrame
df = pd.DataFrame(data)

# Mostrar DataFrame
print(f"DataFrame generado:\n {df}")
# Filtrar personas mayores de 25 años
mayores25 = df[df['Edad'] > 25]

# Imprimir el resultado del filtrado
print(f"Personas mayores de 25 años:\n{mayores25}")

# Agrupar por ciudad y obtener la puntuación máxima en cada ciudad
agrupacion = mayores25.groupby('Ciudad')['Puntuación'].max()

# Imprimir el resultado de la agrupación
print(f"Puntuación máxima por ciudad para personas mayores de 25 años:\n{agrupacion}")
