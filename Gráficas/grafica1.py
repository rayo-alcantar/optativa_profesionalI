import numpy as np
import matplotlib.pyplot as plt

# Gráfico de líneas (comentado como en tu código original)
"""
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(10, 4)) 
plt.plot(x, y)
plt.title("Gráfico de líneas")
plt.xlabel("X")  
plt.ylabel("sin(x)") 
plt.show() 
"""

"""
# Vectores de categorías y valores corregidos
categorias = ['a', 'b', 'c', 'd'] 
valores = [3, 7, 2, 5]

# Crear gráfico de barras
plt.bar(categorias, valores)

# Añadir etiquetas y título
plt.xlabel('Categorías')
plt.ylabel('Valores')
plt.title('Gráfico de Barras')

# Mostrar el gráfico
plt.show()
"""
"""
# Generar vectores aleatorios con 50 elementos
x = np.random.rand(50)
y = np.random.rand(50)

# Crear scatter plot
plt.scatter(x, y)

# Añadir etiquetas y título
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title('Scatter Plot')

# Mostrar el gráfico
plt.show()
"""
# Generar un vector de 1000 datos aleatorios
datos = np.random.randn(1000)

# Crear histograma
plt.hist(datos, bins=30, edgecolor='black')

# Añadir etiquetas y título
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma de 1000 Datos Aleatorios')

# Mostrar el gráfico
plt.show()