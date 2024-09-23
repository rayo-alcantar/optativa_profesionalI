import pandas as pd
import numpy as np

df = pd.DataFrame({
	'A': [1, 2, np.nan, 4, 5],
	'B': [5, 6, 7, np.nan, 9],
	'C': ['a', 'b', 'c', 'd', 'e'],
	'D': ['foo', 'bar', 'foo', 'bar', 'foo']
})

print(df)
df_clean = df.dropna()
print(f'Dataframe sin filas nulas:\n {df_clean}\n')
df_filled = df.fillna(value=df['A'].min())
df_filled = df.fillna(value={'B': df['B'].median()})
print(df_filled)
# Mostrar la columna 'D' con duplicados
print("Columna 'D' con duplicados:")
print(df['D'])

# Eliminar duplicados
df_no_duplicates = df.drop_duplicates()

# Mostrar la columna 'D' sin duplicados
print("\nColumna 'D' sin duplicados:")
print(df_no_duplicates['D'])
# Renombrar las columnas
df_renamed = df.rename(columns={'A': 'Alfa', 'B': 'Beta', 'C': 'Charlie', 'D': 'Delta'})

# Mostrar el DataFrame con las columnas renombradas
print("DataFrame con columnas renombradas:")
print(df_renamed)
