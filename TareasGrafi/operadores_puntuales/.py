import cv2 as cv
import numpy as np

# Crear imagen de prueba (cuadrado gris)
img = np.zeros((300,300), dtype=np.uint8)

# Asignar un cuadrado gris
img[50:250,50:250] = 120

# OPERADOR 1: Negativo
negativo = 255 - img

# OPERADOR 2: Aumentar brillo
brillo = img + 50

# OPERADOR 3: Disminuir brillo
oscuro = img - 50

# OPERADOR 4: Umbralización
_, umbral = cv.threshold(img, 100, 255, cv.THRESH_BINARY)

# OPERADOR 5: Inversión manual
invertida = np.zeros_like(img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        invertida[i,j] = 255 - img[i,j]


# Mostrar resultados
cv.imshow("Original", img)
cv.imshow("Negativo", negativo)
cv.imshow("Brillo aumentado", brillo)
cv.imshow("Brillo disminuido", oscuro)
cv.imshow("Umbral", umbral)
cv.imshow("Invertida manual", invertida)

cv.waitKey(0)
cv.destroyAllWindows()