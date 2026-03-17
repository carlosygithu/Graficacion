import cv2
import numpy as np
import math

# Misión 2: El Código Mareado (Rotación)

# 1. Cargar la imagen
img_qr = cv2.imread('D:\\Proyectos\\GRAFICACION\\TareasGrafi\\qr_rotado.jpg')

if img_qr is None:
    raise FileNotFoundError("Error al cargar la imagen")

# Dimensiones
alto, ancho = img_qr.shape[:2]
print(f"Dimensiones originales: {ancho}x{alto}")

# Centro de la imagen
cx, cy = ancho // 2, alto // 2

# Ángulo de corrección (-45 grados horario)
theta = math.radians(-45)

# MÉTODO 1: RAW (Manual con trigonometría)
canvas = np.zeros_like(img_qr)

for y in range(alto):
    for x in range(ancho):
        # Fórmulas de rotación alrededor del centro
        x_shift = x - cx
        y_shift = y - cy

        x_rot = int(x_shift * math.cos(theta) - y_shift * math.sin(theta) + cx)
        y_rot = int(x_shift * math.sin(theta) + y_shift * math.cos(theta) + cy)

        # Verificar que esté dentro de la imagen
        if 0 <= x_rot < ancho and 0 <= y_rot < alto:
            canvas[y, x] = img_qr[y_rot, x_rot]

cv2.imshow("Rotacion RAW", canvas)

# MÉTODO 2: OpenCV (Optimizado)
M = cv2.getRotationMatrix2D((cx, cy), -45, 1.0)
dst = cv2.warpAffine(img_qr, M, (ancho, alto))

cv2.imshow("Rotacion OpenCV", dst)

# Comparación visual
cv2.waitKey(0)
cv2.destroyAllWindows()
