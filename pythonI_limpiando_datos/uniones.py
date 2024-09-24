import pandas as pd

# Primer DataFrame con ID, Nombre y Departamento
data1 = {
    'id': [1, 2, 3, 4],
    'nombre': ['Ana', 'Juan', 'Maria', 'Carlos'],
    'departamento': ['ventas', 'it', 'marketing', 'rh']
}

df1 = pd.DataFrame(data1)

# Segundo DataFrame con ID y Salario
data2 = {
    'id': [3, 4, 5, 6],
    'salario': [50000, 60000, 55000, 65000]
}

df2 = pd.DataFrame(data2)

# Mostrar los DataFrames
print("Primer DataFrame:")
print(df1)

print("\nSegundo DataFrame:")
print(df2)

# Realizar la unión (merge) basada en la columna 'id' con 'inner'
merged_df = pd.merge(df1, df2, on='id', how='inner')

# Mostrar el DataFrame resultante del primer merge
print("\nDataFrame resultante después del merge (inner join) basado en 'id':")
print(merged_df)

# Realizar la unión (merge) con el error corregido 'how' -> 'outer'
merged_df2 = pd.merge(df1, df2, on='id', how='outer')

# Mostrar el DataFrame resultante del segundo merge
print("\nDataFrame resultante después del merge (outer join) basado en 'id':")
print(merged_df2)

# Realizar la unión (merge) con 'left'
merged_df3 = pd.merge(df1, df2, on='id', how='left')

# Mostrar el DataFrame resultante del tercer merge
print("\nDataFrame resultante después del merge (left join) basado en 'id':")
print(merged_df3)
merged_df_right = pd.merge(df1, df2, on='id', how='right')
print(merged_df_right)
df3 = pd.DataFrame({'id': [7, 8], 'nombre': ['Sofía', 'Pedro'], 'departamento': ['finanzas', 'ventas']})
concatenated_df = pd.concat([df1, df3])
print(concatenated_df)
concatenated_df = pd.concat([df1, df2])
print(concatenated_df)
