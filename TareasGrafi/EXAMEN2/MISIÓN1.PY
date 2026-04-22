
import cv2
import numpy as np

# Cargar la imagen oscura
img = cv2.imread('D:\\Proyectos\\GRAFICACION\\TareasGrafi\\EXAMEN2\\m1_oscura 1.png', cv2.IMREAD_GRAYSCALE)

# -------------------------------
# MODO RAW (ciclos anidados)
# -------------------------------
recuperado_x50 = np.zeros_like(img, dtype=np.int32)

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        valor = img[y, x] * 50
        recuperado_x50[y, x] = np.clip(valor, 0, 255)

# Convertir a uint8 y guardar primera evidencia
recuperado_x50 = recuperado_x50.astype(np.uint8)
cv2.imwrite("m1_recuperado_x50.png", recuperado_x50)

# Segunda fase RAW (+20)
recuperado_x50_mas20 = np.zeros_like(img, dtype=np.int32)
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        valor = img[y, x] * 50 + 20
        recuperado_x50_mas20[y, x] = np.clip(valor, 0, 255)

recuperado_x50_mas20 = recuperado_x50_mas20.astype(np.uint8)
cv2.imwrite("m1_recuperado_x50_mas20.png", recuperado_x50_mas20)

# -------------------------------
# MODO VECTORIZADO (NumPy/OpenCV)
# -------------------------------
# Primera fase
recuperado_numpy_x50 = np.clip(img * 50, 0, 255).astype(np.uint8)
cv2.imwrite("m1_recuperado_x50_numpy.png", recuperado_numpy_x50)

# Segunda fase (+20)
recuperado_numpy_x50_mas20 = np.clip(img * 50 + 20, 0, 255).astype(np.uint8)
cv2.imwrite("m1_recuperado_x50_mas20_numpy.png", recuperado_numpy_x50_mas20)
