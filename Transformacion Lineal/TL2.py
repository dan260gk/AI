import cv2

# Cargar la imagen
imagen = cv2.imread('lucialotus.jpg')

# Escalar la imagen al 50% del tamaño original
nueva_imagen = cv2.resize(imagen, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# Mostrar la imagen original y la escalada
cv2.imshow('Imagen Original', imagen)
cv2.imshow('Imagen Escalada', nueva_imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
