import cv2
import numpy as np
 
# Cargar la imagen
imagen = cv2.imread('ardilla.jpg')
 
# Definir la matriz de transformación para una translación de 50 píxeles hacia la derecha y 30 píxeles hacia abajo
matriz_translacion = np.float32([[1, 0, 50], [0, 1, 30]])
 
# Aplicar la transformación lineal a la imagen
imagen_transformada = cv2.warpAffine(imagen, matriz_translacion, (imagen.shape[1], imagen.shape[0]))
 
# Mostrar la imagen original
cv2.imshow('Imagen Original', imagen)
 
# Mostrar la imagen transformada
cv2.imshow('Imagen Transformada', imagen_transformada)
 
# Esperar a que se presione una tecla y luego cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()