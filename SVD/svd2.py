# Paso 1: Importar las bibliotecas necesarias
from numpy import array, diag, zeros
from scipy.linalg import svd
# Paso 2: Crear un conjunto de datos de ejemplo
datos = array([[100, 8, 7],
               [80, 5, 9],
               [120, 10, 5]])

# Paso 4: Aplicar SVD
U, s, VT = svd(datos)
# Paso 5: Selección de componentes principales
num_componentes = 2
Sigma = zeros((datos.shape[0], datos.shape[1]))
Sigma[:datos.shape[0], :datos.shape[0]] = diag(s)
Sigma = Sigma[:, :num_componentes]
componentes_principales = VT[:num_componentes, :]

print("Matriz U")
print(U)
print("Matriz Sigma")
print(Sigma)
print("Matriz VT")
print(VT)
VT = VT[:num_componentes, :]
print("Datos originales")
print(datos)
# Paso 6: Proyección de datos en el nuevo espacio dimensional
print("Datos reducidos 1")
datos_reducidos = U.dot(Sigma)
print(datos_reducidos)
print("Datos reducidos 2")
datos_reducidos = datos.dot(VT.T)
print(datos_reducidos)
# Reconstrucción de los datos originales
datos_reconstruidos = U.dot(Sigma.dot(VT))
print("Reconstrucción")
print(datos_reconstruidos)