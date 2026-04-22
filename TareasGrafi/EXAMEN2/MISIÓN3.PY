
import cv2
import numpy as np
import math

# Crear lienzo 600x600 con color base BGR(40,20,20)
img = np.zeros((600, 600, 3), dtype=np.uint8)
img[:] = (40, 20, 20)

# -------------------------------
# Círculos amarillos
# -------------------------------
cv2.circle(img, (300, 300), 170, (0, 255, 255), 3)  # exterior
cv2.circle(img, (300, 300), 110, (0, 255, 255), 2)  # interior

# -------------------------------
# Rectángulo rojo sólido
# -------------------------------
cv2.rectangle(img, (250, 260), (350, 340), (0, 0, 255), -1)

# -------------------------------
# Líneas diagonales blancas (X)
# -------------------------------
cv2.line(img, (0, 0), (600, 600), (255, 255, 255), 2)
cv2.line(img, (600, 0), (0, 600), (255, 255, 255), 2)

# -------------------------------
# Marca simétrica: 8 círculos verdes
# -------------------------------
radio = 140
for i in range(8):
    ang = i * (360 / 8)
    rad = math.radians(ang)
    x = int(300 + radio * math.cos(rad))
    y = int(300 + radio * math.sin(rad))
    cv2.circle(img, (x, y), 8, (0, 255, 0), -1)

# -------------------------------
# Texto blanco SECTOR-9
# -------------------------------
cv2.putText(img, "SECTOR-9", (200, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

# -------------------------------
# Guardar resultado
# -------------------------------
cv2.imwrite("m3_sello_forjado_v2.png", img)
