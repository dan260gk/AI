import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('lucialotus.jpg')

# Definir la matriz de transformación
# escalará la imagen en un factor de 1.5
matrix = np.float32([[1.5, 0, 0],
                     [0, 1.5, 0]])

# Aplicar la transformación lineal
transformed_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

# Mostrar la imagen original y la transformada
cv2.imshow('Original', image)
cv2.imshow('Transformada', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
