import cv2
import numpy as np

# Misión 1: El Artefacto Desplazado (Traslación)

# 1. Cargar la imagen original
img = cv2.imread('D:\\Proyectos\\GRAFICACION\\TareasGrafi\\vehiculo.jpg')

# Verificar que la imagen se cargó
if img is None:
    raise FileNotFoundError("Error al cargar la imagen")

# Dimensiones originales
alto, ancho = img.shape[:2]
print(f"Dimensiones originales: {ancho}x{alto}")

# MÉTODO 1: RAW (Manual con NumPy)
# Crear un lienzo negro del mismo tamaño
canvas = np.zeros((alto, ancho, 3), dtype=np.uint8)

# Traslación: 300 en X y 200 en Y
tx, ty = 300, 200

# Copiar píxeles aplicando traslación
# Nota: se recorta para no salir del rango
canvas[ty:alto, tx:ancho] = img[0:alto-ty, 0:ancho-tx]

# Mostrar resultado
cv2.imshow("Traslacion RAW", canvas)

# MÉTODO 2: OpenCV (Optimizado)
# Matriz de traslación
M = np.float32([[1, 0, tx],
                [0, 1, ty]])

# Aplicar warpAffine
dst = cv2.warpAffine(img, M, (ancho, alto))

# Mostrar resultado
cv2.imshow("Traslacion OpenCV", dst)

# Comparación visual
cv2.waitKey(0)
cv2.destroyAllWindows()
