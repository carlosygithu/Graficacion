import cv2 as cv
import numpy as np

# Crear imagen base
img = np.zeros((300,300), dtype=np.uint8)

# Dibujar un cuadrado blanco
img[100:200,100:200] = 255

# Obtener dimensiones
filas, columnas = img.shape

# 1. TRASLACION
M_traslacion = np.float32([[1,0,50],[0,1,50]])
traslacion = cv.warpAffine(img, M_traslacion, (columnas, filas))

# 2. ROTACION
M_rotacion = cv.getRotationMatrix2D((columnas/2, filas/2), 45, 1)
rotacion = cv.warpAffine(img, M_rotacion, (columnas, filas))

# 3. ESCALADO
escalado = cv.resize(img, None, fx=1.5, fy=1.5, interpolation=cv.INTER_NEAREST)

# 4. CIZALLAMIENTO
M_cizalla = np.float32([[1,0.5,0],[0,1,0]])
cizalla = cv.warpAffine(img, M_cizalla, (columnas, filas))

# 5. REFLEXION
reflexion = cv.flip(img, 1)

# Mostrar resultados
cv.imshow("Original", img)
cv.imshow("Traslacion", traslacion)
cv.imshow("Rotacion", rotacion)
cv.imshow("Escalado", escalado)
cv.imshow("Cizallamiento", cizalla)
cv.imshow("Reflexion", reflexion)

cv.waitKey(0)
cv.destroyAllWindows()