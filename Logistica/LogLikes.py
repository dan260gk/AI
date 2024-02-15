# Importar la biblioteca numpy con el alias np
import numpy as np

# Importar la función train_test_split desde el módulo model_selection de sklearn
# Esta función se utiliza para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Importar la clase LogisticRegression desde el módulo linear_model de sklearn
# Esta clase implementa la regresión logística, un método de clasificación
from sklearn.linear_model import LogisticRegression

# Importar la función accuracy_score desde el módulo metrics de sklearn
# Esta función se utiliza para calcular la precisión de un modelo de clasificación
from sklearn.metrics import accuracy_score


# Datos obtenidos de una cuenta de TikTok
data = {
    'Date': ['2023-12-16', '2023-12-17', '2023-12-18', '2023-12-19', '2023-12-20', '2023-12-21', '2023-12-22', '2023-12-23', '2023-12-24', '2023-12-25', '2023-12-26', '2023-12-27', '2023-12-28', '2023-12-29', '2023-12-30', '2023-12-31', '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12', '2024-01-13', '2024-01-14', '2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20', '2024-01-21', '2024-01-22', '2024-01-23', '2024-01-24', '2024-01-25', '2024-01-26', '2024-01-27', '2024-01-28', '2024-01-29', '2024-01-30', '2024-01-31', '2024-02-01', '2024-02-02', '2024-02-03', '2024-02-04', '2024-02-05', '2024-02-06', '2024-02-07', '2024-02-08', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13'],
    'Video_Views': [46, 87, 674, 59, 250, 54, 58, 239, 73, 46, 49, 42, 29, 45, 37, 45, 29, 29, 50, 19, 44, 31, 25, 22, 34, 43, 32, 148, 402, 921, 212, 343, 106, 362, 104, 190, 162, 131, 60, 108, 142, 61, 73, 99, 90, 51, 48, 22, 43, 72, 52, 47, 37, 40, 29, 27, 46, 47, 48, 42],
    'Profile_Views': [1, 0, 3, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 3, 0, 1, 1, 1, 1, 1, 0,  2, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,0 ,0 ,0 ,1, 0],
    'Likes': [1, 2, 14, 1, 5, 1, 1, 8, 0, 0, 0, 2, 1, 1, 1, 2, 0, 1, 0, 0, 0,1, 0, 0, 0, 2, 1, 6, 15, 20, 5, 15, 0, 24, 7, 7, 7, 6, 2, 1, 3, 1, 1, 0, 1, 3, 0, 0, 1, 0, 1,0,1, 1, 2, 0, 0, 0, 0, 0],
    'Comments': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
    'Shares': [0, 0, 1, 0, 2, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
    'Unique_Viewers': [36, 41, 505, 33, 204, 40, 38, 176, 45, 37, 34, 31, 24, 28, 27, 36, 23, 25, 39, 14, 29, 22, 22, 19, 24, 31, 26, 116, 298, 748, 144, 288, 78, 303, 78, 125, 127, 77, 45, 60, 107, 45, 54, 68, 61, 37, 41, 17, 31, 44, 34, 40, 26, 26, 26, 21, 35, 35, 38, 31],
}

# Crear una variable binaria para likes > 5
likes_over_5 = [1 if like > 5 else 0 for like in data['Likes']]
# Verificar la longitud de cada columna //ignorar, solo se uso para verificar
"""
print("Longitud de Video_Views:", len(data['Video_Views']))
print("Longitud de Profile_Views:", len(data['Profile_Views']))
print("Longitud de Comments:", len(data['Comments']))
print("Longitud de Shares:", len(data['Shares']))
print("Longitud de Unique_Viewers:", len(data['Unique_Viewers']))
"""
# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    #caracteristicas de entrada
    np.column_stack((data['Video_Views'], data['Profile_Views'], data['Comments'], data['Shares'], data['Unique_Viewers'])), 
    likes_over_5, #Etiquetas de salida correspondientes a si la cantidad de "Likes" supera 5 (0 si no, 1 si sí).
    test_size=1, # Tamaño del conjunto de prueba, 100% del tamaño total del conjunto de datos.
    random_state=42 # Fijamos la semilla aleatoria para asegurar reproducibilidad en la división de los datos.
)

# Crear el modelo
modelo = LogisticRegression()

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", precision*100,"%")
# + codigo para hacer pruebas
'''
print("resultados del modelo de regresión logística:\n")
print("El intercepto es:", modelo.intercept_[0])
print("Los coeficientes son:", modelo.coef_[0])
'''
print("Para calcular la probabilidad de que un video tenga mas de 5 likes se utiliza la siguiente formula:\n")
print("p(y = 1 | x) = 1 / (1 + exp(-(b0 + b1x1 + b2x2 + b3x3 + b4x4 + b5x5)))\n")
print("Donde:")
print("b0 es el intercepto ({:.2f}).".format(modelo.intercept_[0]))
print("b1 es el coeficiente del número de vistas de video ({:.2f}).".format(modelo.coef_[0][0]))
print("b2 es el coeficiente del número de visitas de perfil ({:.2f}).".format(modelo.coef_[0][1]))
print("b3 es el coeficiente del número de comentarios ({:.2f}).".format(modelo.coef_[0][2]))
print("b4 es el coeficiente del número de compartidos ({:.2f}).".format(modelo.coef_[0][3]))
print("b5 es el coeficiente del número de espectadores únicos ({:.2f}).".format(modelo.coef_[0][4]))
print("Xn son los datos que se agregaran del mismo tipo del coeficiente")
# Ejemplo de cálculo de probabilidad individual
print("\nEntonces la formula quedaria de la siguiente manera:\n")
print("p(y = 1 | x) = 1 / (1 + exp(-({:.2f} + {:.2f}*Video_Views + {:.2f}*Profile_Views + {:.2f}*Comments + {:.2f}*Shares + {:.2f}*Unique_Viewers)))".format(modelo.intercept_[0], modelo.coef_[0][0], modelo.coef_[0][1], modelo.coef_[0][2], modelo.coef_[0][3], modelo.coef_[0][4]))

# Valores de ejemplo para las características del caso específico
ejemplo_video_views = 100
ejemplo_profile_views = 2
ejemplo_comments = 1
ejemplo_shares = 0
ejemplo_unique_viewers = 50

# Calcular la probabilidad individual utilizando la ecuación logística
probabilidad = 1 / (1 + np.exp(-(modelo.intercept_[0] + 
                                   modelo.coef_[0][0] * ejemplo_video_views + 
                                   modelo.coef_[0][1] * ejemplo_profile_views + 
                                   modelo.coef_[0][2] * ejemplo_comments + 
                                   modelo.coef_[0][3] * ejemplo_shares + 
                                   modelo.coef_[0][4] * ejemplo_unique_viewers)))
print("\nPor ejemplo, para un caso con:")
print("Número de vistas de video: {}".format(ejemplo_video_views))
print("Número de vistas de perfil: {}".format(ejemplo_profile_views))
print("Número de comentarios: {}".format(ejemplo_comments))
print("Número de compartidos: {}".format(ejemplo_shares))
print("Número de espectadores únicos: {}".format(ejemplo_unique_viewers))
print("\nLa probabilidad de que el video tenga mas de 5 likes es: {:.2f}%".format(probabilidad * 100))