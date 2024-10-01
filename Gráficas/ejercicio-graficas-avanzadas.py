# Autor: Ángel De Jesús Alcántar
# Descripción: Este script analiza el dataset 'student_performance.csv' que contiene información de estudiantes
# y realiza diferentes visualizaciones, incluyendo gráficos interactivos y no interactivos.
# Dataset obtenido desde: https://github.com/rayo-alcantar/optativa_profesionalI/blob/main/datos/student_performance.csv

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset desde GitHub
url = "https://raw.githubusercontent.com/rayo-alcantar/optativa_profesionalI/main/datos/student_performance.csv"
df = pd.read_csv(url)

# 2. Explicación de cada variable:
# - StudentID: Identificación única para cada estudiante.
# - Name: Nombre del estudiante.
# - Gender: Género del estudiante (Male/Female).
# - AttendanceRate: Tasa de asistencia en porcentaje.
# - StudyHoursPerWeek: Horas que el estudiante dedica al estudio por semana.
# - PreviousGrade: Nota del estudiante en el periodo anterior.
# - ExtracurricularActivities: Cantidad de actividades extracurriculares.
# - ParentalSupport: Nivel de apoyo parental (High, Medium, Low).
# - FinalGrade: Nota final del estudiante.

# Gráfico 1: Gráfico de dispersión interactivo - Horas de estudio vs Calificación final
# En este gráfico visualizamos cómo varían las calificaciones en función de las horas de estudio,
# con el tamaño del punto representando la cantidad de actividades extracurriculares.

fig_scatter = px.scatter(df, x='StudyHoursPerWeek', y='FinalGrade', size='ExtracurricularActivities', color='Gender',
                         hover_data=['AttendanceRate', 'PreviousGrade'],
                         title="Gráfico de Dispersión Interactivo: Horas de Estudio vs Calificación Final")
fig_scatter.show()

# Gráfico 2: Gráfico de líneas interactivo - Tasa de asistencia y calificaciones
# En este gráfico mostramos cómo la tasa de asistencia afecta tanto la calificación final como la calificación previa.
x_vals = df['AttendanceRate']
y_vals1 = df['FinalGrade']
y_vals2 = df['PreviousGrade']

fig_lines = go.Figure()
fig_lines.add_trace(go.Scatter(x=x_vals, y=y_vals1, mode='lines+markers', name='Calificación Final'))
fig_lines.add_trace(go.Scatter(x=x_vals, y=y_vals2, mode='lines+markers', name='Calificación Previa'))

fig_lines.update_layout(title='Gráfico de Líneas Interactivo: Asistencia vs Calificaciones',
                        xaxis_title='Tasa de Asistencia (%)', yaxis_title='Calificación')
fig_lines.show()

# Gráfico 3: Gráfico de barras - Actividades extracurriculares por estudiante
# Este gráfico de barras nos muestra la cantidad de actividades extracurriculares que realizan los estudiantes.

plt.figure(figsize=(10, 6))
sns.barplot(x='Name', y='ExtracurricularActivities', data=df)
plt.title('Actividades Extracurriculares por Estudiante')
plt.xticks(rotation=45)
plt.show()

# Gráfico 4: Gráfico de dispersión - Asistencia vs Calificación Final
# Este gráfico nos permite ver si existe alguna relación entre la tasa de asistencia y la calificación final.

plt.figure(figsize=(10, 6))
sns.scatterplot(x='AttendanceRate', y='FinalGrade', hue='Gender', data=df)
plt.title('Asistencia vs Calificación Final')
plt.show()

# Gráfico 5: Histograma - Distribución de las calificaciones finales
# Un histograma que muestra la distribución de las calificaciones finales obtenidas por los estudiantes.

plt.figure(figsize=(10, 6))
sns.histplot(df['FinalGrade'], kde=True)
plt.title('Distribución de las Calificaciones Finales')
plt.show()
