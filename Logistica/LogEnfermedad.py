from sklearn.linear_model import LogisticRegression
import numpy as np

# Datos de entrenamiento
X_train = np.array([[20, 1], [30, 0], [40, 1], [50, 0]])  # Edad y sexo
y_train = np.array([0, 1, 1, 0])  # 0 para "No", 1 para "Sí" (enfermedad)

# Creación del modelo
modelo = LogisticRegression()

# Entrenamiento del modelo
modelo.fit(X_train, y_train)

# Parámetros aprendidos por el modelo
intercepto = modelo.intercept_[0]
coef_edad = modelo.coef_[0][0]
coef_sexo = modelo.coef_[0][1]

# Edad y sexo del paciente
edad_paciente = 40
sexo_paciente = 1  # 1 para hombre, 0 para mujer

# Calculando la probabilidad individual
z = intercepto + coef_edad * edad_paciente + coef_sexo * sexo_paciente
probabilidad = 1 / (1 + np.exp(-z))

print("El intercepto es:", intercepto)
print("El coeficiente de la edad es:", coef_edad)
print("El coeficiente del sexo es:", coef_sexo)

print("\nLa probabilidad de que un hombre de 40 años tenga la enfermedad es aproximadamente:", probabilidad)

