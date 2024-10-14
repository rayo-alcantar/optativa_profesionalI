import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
"""
Aquí se carga el conjunto de datos de casas de Nueva York y se realiza una exploración inicial:
1. Se muestra las primeras filas del DataFrame
2. Se obtiene información sobre las columnas y tipos de datos
 Se generan estadísticas descriptivas de las variables numéricas"""
"""
#Cargamos la bd
data = pd.read_csv('https://github.com/rayo-alcantar/optativa_profesionalI/blob/main/datos/NY-House-Dataset.csv',sep = ',')#
#data = pd.read_csv("C:/Users/Lizeth Solano Romo/OneDrive - Universidad Autónoma de Aguascalientes/2024/MATERIAS/agosto-diciembre/LITC/7o/Ejercicios Phyton/NY-House-Dataset.csv")


data.head()

data.info()

# Realizamos el analisis exploratorio:
##Conteo de categorias

print(data["BROKERTITLE"].value_counts())
print(data["TYPE"].value_counts())
print(data["STATE"].value_counts())
print(data["LOCALITY"].value_counts())
print(data["PROPERTYSQFT"].value_counts())
print(data["SUBLOCALITY"].value_counts())

##descripcion de variables numericas:

print(data[["PRICE","BEDS","BATH","PROPERTYSQFT"]].describe())

#Transformar categorias con menos de 10 registros a una nueva:
## BROKERTITLE,TYPE,STATE,CP,STATE_2,CODIGO_POSTAL,LOCALLITY, SUBLOCALLITY

BROKERTITLE_conteo = data["BROKERTITLE"].value_counts()
BROKERTITLE_conservar = BROKERTITLE_conteo[BROKERTITLE_conteo>=10].index.to_list()
data["BROKERTITLE"]=np.where(data["BROKERTITLE"].isin(BROKERTITLE_conservar),data["BROKERTITLE"],"No se especifica")

#TYPE
TYPE_conteo = data["TYPE"].value_counts()
TYPE_conservar = TYPE_conteo[TYPE_conteo>=10].index.to_list()
data["TYPE"]=np.where(data["TYPE"].isin(TYPE_conservar),data["TYPE"],"No se especifica")

#STATE Y CP
data["STATE_2"]=data["STATE"].str.extract("^([^,]+)")
data["codigo_postal"]=data["STATE"].str.extractall("(\d+)").groupby(level=0).apply(lambda x: x.iloc[-1])

#STATE_2
STATE_2_conteo = data["STATE_2"].value_counts()
STATE_2_conservar = STATE_2_conteo[STATE_2_conteo>=10].index.to_list()
data["STATE_2"]=np.where(data["STATE_2"].isin(STATE_2_conservar),data["STATE_2"],"No se especifica")

#codigo_postal
codigo_postal_conteo = data["codigo_postal"].value_counts()
codigo_postal_conservar = codigo_postal_conteo[codigo_postal_conteo>=10].index.to_list()
data["codigo_postal"]=np.where(data["codigo_postal"].isin(codigo_postal_conservar),data["codigo_postal"],"No se especifica")

#LOCALITY
LOCALITY_conteo = data["LOCALITY"].value_counts()
LOCALITY_conservar = LOCALITY_conteo[LOCALITY_conteo>=10].index.to_list()
data["LOCALITY"]=np.where(data["LOCALITY"].isin(LOCALITY_conservar),data["LOCALITY"],"No se especifica")


#SUBLOCALITY
SUBLOCALITY_conteo = data["SUBLOCALITY"].value_counts()
SUBLOCALITY_conservar = SUBLOCALITY_conteo[SUBLOCALITY_conteo>=10].index.to_list()
data["SUBLOCALITY"]=np.where(data["SUBLOCALITY"].isin(SUBLOCALITY_conservar),data["SUBLOCALITY"],"No se especifica")

#TABLA DESCRIPTIVA:

Tabla_Descriptiva_Final = data.groupby(['TYPE','STATE_2','codigo_postal','LOCALITY','SUBLOCALITY']).agg(
  CONTEO=('TYPE','count'),
  PROMEDIO_PRICE=('PRICE','mean'),
  DESVIACION_PRICE=('PRICE','std'),
  PROMEDIO_BEDS=('BEDS','mean'),
  DESVIACION_BEDS=('BEDS','std'),
  PROMEDIO_BATH=('BATH','mean'),
  DESVIACION_BATH=('BATH','std'),
  PROMEDIO_PROPERTYSQFT=('PROPERTYSQFT','mean'),
  DESVIACION_PROPERTYSQFT=('PROPERTYSQFT','std')
).reset_index()

print(Tabla_Descriptiva_Final)
#Elimina los datos nulos de la base de datos

data = data.dropna()


#Modelo


### Gráfico de dispersión que muestra la relación entre el tamaño y el precio de venta

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PROPERTYSQFT', y='PRICE', data=df)
plt.title('Relación entre Tamaño y Precio de Venta')
plt.xlabel('Tamaño')
plt.ylabel('Precio')
plt.show()
"""
Se propone un modelo y su evaluación:
1. Se divide el conjunto de datos en entrenamiento y prueba.
2. Se crea y entrena un modelo de regresión lineal.
3. Se hacen predicciones y se evalúa el rendimiento del modelo.
La **regresión lineal** es una técnica estadística fundamental utilizada para modelar la relación entre variables y hacer predicciones.

Es un método estadístico que busca establecer una relación lineal entre una variable dependiente (Y) y una o más variables independientes (X). En su forma más simple, se representa mediante la ecuación:

#Y = β₀ + β₁X + ε

Donde:

#β₀ es la intersección con el eje Y

#β₁ es la pendiente de la línea

#ε es el término de error


Sirve para:

- *Predicción:* Permite hacer pronósticos sobre la variable dependiente basándose en los valores de las variables independientes.
- *Análisis de relaciones:* Ayuda a entender cómo cambia la variable dependiente cuando se modifican las variables independientes.
- *Identificación de tendencias:* Permite detectar patrones y tendencias en los datos.
- *Cuantificación de impactos:* Mide el efecto que tienen las variables independientes sobre la variable dependiente.

*En qué casos se utiliza*

La regresión lineal se aplica en diversos campos y situaciones, como:
- Economía y finanzas: Para predecir ventas, analizar el impacto de variables económicas, o estimar el valor de activos.
- Ciencias sociales: En estudios que buscan relacionar factores socioeconómicos con diversos resultados.
- Medicina: Para analizar la relación entre dosis de medicamentos y respuestas fisiológicas.
- Marketing: En la predicción de comportamientos de consumo o efectividad de campañas publicitarias.
- Ingeniería: Para modelar relaciones entre variables en sistemas físicos.
- Ciencias ambientales: En el estudio de fenómenos climáticos o ecológicos.
- Recursos humanos: Para analizar factores que influyen en el desempeño o satisfacción laboral.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Error cuadrático medio: {mse}')
print(f'R-cuadrado: {r2}')
"""
Interpretando:
- Error cuadrático medio (MSE): 19243479524898.84
El MSE mide el promedio de los errores al cuadrado entre los valores predichos y los valores reales. Un valor más bajo indica mejores predicciones.
En este caso, el MSE es bastante alto, lo que sugiere que hay una discrepancia significativa entre los precios predichos y los reales. Sin embargo, es importante notar que el MSE está en la misma escala que el cuadrado de los precios de las casas, por lo que un número grande no es necesariamente inusual para datos de precios de propiedades.
#- R-cuadrado (R²): 0.23621329447296957
#El R² indica qué proporción de la varianza en la variable dependiente (precio de venta) es predecible a partir de las variables independientes (características de la casa). Varía de 0 a 1, donde 1 indica una predicción perfecta.
#Un R² de aproximadamente 0.236 significa que el modelo explica alrededor del 23.6% de la variabilidad en los precios de las casas. Esto sugiere que:
1. El modelo tiene cierto poder predictivo, ya que explica más del 0% de la varianza.
2. Sin embargo, hay una gran parte de la variabilidad (aproximadamente el 76.4%) que el modelo no explica.

*Interpretación general:* Estos resultados indican que el modelo tiene un poder predictivo limitado. Aunque puede capturar algunas tendencias en los datos, hay muchos factores que influyen en los precios de las casas que no están siendo considerados o que no se están modelando adecuadamente con una regresión lineal simple.
"""

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Precio real')
plt.ylabel('Precio predicho')
plt.title('Comparación entre precios reales y predichos')
plt.show()

## Posible explicación:

"""
- Puede haber relaciones no lineales entre las variables que una regresión lineal no captura.
- Podrían faltar variables importantes que influyen en el precio de las casas.
- Podría haber outliers o datos ruidosos que afectan el rendimiento del modelo.

## Pasos a seguir:
1. Explorar más a fondo los datos para entender mejor las relaciones entre variables.
2. Considerar la inclusión de más características relevantes si están disponibles.
3. Probar técnicas de preprocesamiento como la normalización o la eliminación de outliers.
4. Experimentar con modelos más complejos que puedan capturar relaciones no lineales.

# OTRO MODELO

Seleccionar características relevantes

# Random Forest

## Qué es

*Random Forest* es un algoritmo de aprendizaje supervisado que pertenece a la familia de métodos de conjunto (ensemble). Funciona creando múltiples árboles de decisión y combinando sus resultados para obtener una predicción más precisa y estable.

## Para qué sirve

1. Predicción: Puede utilizarse tanto para problemas de clasificación como de regresión.

2. Selección de características: Ayuda a identificar las variables más importantes en un conjunto de datos.

3. Manejo de datos faltantes: Es capaz de manejar eficientemente conjuntos de datos con valores faltantes.

4. Reducción del sobreajuste: Al combinar múltiples árboles, reduce el riesgo de sobreajuste común en los árboles de decisión individuales.

5. Estimación de la importancia de variables: Proporciona una medida de la importancia relativa de cada característica en la predicción.

## En qué casos se utiliza

Random Forest se aplica en diversos campos y situaciones, como:

1. Finanzas:
   - Detección de fraudes bancarios
   - Predicción de riesgos crediticios
   - Análisis de mercados financieros

2. Medicina y salud:
   - Diagnóstico de enfermedades
   - Predicción de la sensibilidad a medicamentos
   - Análisis de imágenes médicas

3. Marketing y comercio electrónico:
   - Predicción del comportamiento de los clientes
   - Segmentación de mercado
   - Recomendación de productos

4. Ciencias ambientales:
   - Predicción del clima
   - Análisis de patrones ecológicos
   - Evaluación de riesgos ambientales

5. Reconocimiento de imágenes y voz:
   - Clasificación de imágenes
   - Reconocimiento de patrones en señales de audio

6. Recursos humanos:
   - Predicción de la rotación de empleados
   - Evaluación del desempeño laboral

7. Industria y manufactura:
   - Predicción de fallos en equipos
   - Optimización de procesos de producción

*Random Forest* es especialmente útil cuando se trabaja con conjuntos de datos grandes y complejos, con muchas variables y posibles interacciones entre ellas. Su capacidad para manejar tanto variables numéricas como categóricas, así como su robustez frente al ruido en los datos, lo hacen una opción popular en muchos campos de aplicación.
"""
