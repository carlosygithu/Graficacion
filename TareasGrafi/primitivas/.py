import cv2 as cv
import numpy as np 

# Crea una imagen de 500x500 píxeles con valor 150 (color gris)
img = np.ones((500,500), np.uint8)*150 

# Dibuja un rectángulo
cv.rectangle(img, (10,10), (200,100), (34,56,100), -1)

# Dibuja un círculo pequeño en el centro
cv.circle(img, (255,255), 1, (23, 43, 144), -1 )

# Dibuja una línea
cv.line(img, (255,255), (200,100), (23, 244, 144), 4)

# Ciclo que repite 400 veces para crear animación
for i in range(400):

    # Dibuja un círculo creciente
    cv.circle(img, (i,i), i , (255, 0, 0), -1 )

    # Dibuja un rectángulo en movimiento
    cv.rectangle(img, (10+i,10), (200,100), (34,56,100), -1)

    # Muestra la imagen
    cv.imshow('img', img)

    # Espera 30 milisegundos
    cv.waitKey(30)
    
# Muestra la imagen final
cv.imshow('img', img)

# Espera una tecla
cv.waitKey(0)

# Cierra ventanas
cv.destroyAllWindows()