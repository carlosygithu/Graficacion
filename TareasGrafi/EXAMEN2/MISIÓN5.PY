
import cv2
import numpy as np

# -------------------------------
# 1. Generar imagen con ruido
# -------------------------------
img = np.random.randint(0, 256, (300, 700, 3), dtype=np.uint8)

# Escribir texto con "tinta tramposa":
# Canal G muy alto, canal B bajo
texto = "HUELLA_CANAL: SECTOR-5"
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, texto, (50, 150), font, 1.5, (0, 255, 50), 3, cv2.LINE_AA)

# Guardar evidencia inicial
cv2.imwrite('D:\\Proyectos\\GRAFICACION\\TareasGrafi\\EXAMEN2\\m5_tricolor.png', img)

# -------------------------------
# 2. Recuperación del mensaje
# -------------------------------
b, g, r = cv2.split(img)

# Probar combinaciones
abs_gb = cv2.absdiff(g, b)

# Normalizar para mejor contraste
recuperado = cv2.normalize(abs_gb, None, 0, 255, cv2.NORM_MINMAX)

# Guardar mejor recuperación
cv2.imwrite("m5_mensaje.png", recuperado)
