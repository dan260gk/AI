import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Definimos la función objetivo
def f(x):
    return x**2 + 3*x + 2

# Creamos un tensor de PyTorch para representar el parámetro x
x = torch.tensor([0.0], requires_grad=True)

# Definimos el optimizador (descenso de gradiente) y la tasa de aprendizaje
optimizer = optim.SGD([x], lr=0.1)

# Realizamos el proceso de optimización durante 100 iteraciones
for _ in range(100):
    # Calculamos el valor de la función
    output = f(x)
    
    # Hacemos el backpropagation (cálculo de gradientes)
    optimizer.zero_grad()
    output.backward()
    
    # Actualizamos los parámetros utilizando el descenso de gradiente
    optimizer.step()

# Imprimimos el valor óptimo de x y el mínimo de la función
print("Valor óptimo de x:", x.item())
print("Mínimo de la función:", f(x).item())

# Creamos un rango de valores de x para graficar la función
x_values = np.linspace(-4, 2, 100)
y_values = f(torch.tensor(x_values)).numpy()

# Graficamos la función
plt.plot(x_values, y_values, label='f(x) = x^2 + 3x + 2')
plt.scatter(x.item(), f(x).item(), color='red', label='Mínimo')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Función y mínimo')
plt.legend()
plt.grid(True)
plt.show()
