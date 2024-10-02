# Ángel De Jesús Alcantar Garza
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Cargar el dataset desde GitHub
url = "https://raw.githubusercontent.com/rayo-alcantar/optativa_profesionalI/main/datos/Netflix.csv"
df = pd.read_csv(url)

# 2. Mostrar la estructura del DataFrame
#print(df.dtypes)

try:
    # Intentar convertir la columna 'date_added' a formato datetime
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    print("Columna 'date_added' convertida a datetime correctamente.")
except Exception as e:
    print(f"Error al convertir 'date_added': {e}")
    try:
        # Si hay error, aplicar un formato específico
        df['date_added'] = pd.to_datetime(df['date_added'], format='%d%%y', errors='coerce')
        print("Columna 'date_added' convertida con formato '%d%%y'.")
    except Exception as e:
        print(f"Error al aplicar formato personalizado: {e}")
#print(df['date_added'].dtype)
print(df['date_added'].head())
#print(df['date_added'].isna().sum())
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
df['rating_numeric'] = df['rating'].map(rating_map)

# Verificar las primeras filas para confirmar
print(df[['rating', 'rating_numeric']].head())
