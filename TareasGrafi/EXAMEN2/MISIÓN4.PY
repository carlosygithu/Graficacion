
import cv2
import numpy as np

# Cargar la imagen con ruido
img = cv2.imread('D:\\Proyectos\\GRAFICACION\\TareasGrafi\\EXAMEN2\\m4_ruido 1.png')

# -------------------------------
# 1. Suavizado con kernel promedio 3x3
# -------------------------------
kernel = np.ones((3, 3), np.float32) / 9
suavizada = cv2.filter2D(img, -1, kernel)
cv2.imwrite("m4_suavizada.png", suavizada)

# -------------------------------
# 2. Conversión a HSV
# -------------------------------
hsv = cv2.cvtColor(suavizada, cv2.COLOR_BGR2HSV)

# -------------------------------
# 3. Segmentación del color Cyan
# Hue ~ 90 → definimos un rango
# -------------------------------
low_cyan = np.array([80, 100, 100])
high_cyan = np.array([100, 255, 255])

mask_cyan = cv2.inRange(hsv, low_cyan, high_cyan)

# -------------------------------
# 4. Guardar resultados
# -------------------------------
cv2.imwrite("m4_mask_cyan.png", mask_cyan)
