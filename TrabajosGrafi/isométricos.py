import cv2
import numpy as np
import math

# --- Función de proyección isométrica ---
def iso_transform(x, y, z=0):
    # Ángulos típicos de proyección isométrica
    angle_x = math.radians(30)
    angle_y = math.radians(30)
    iso_x = (x - y) * math.cos(angle_x)
    iso_y = (x + y) * math.sin(angle_y) - z
    return int(iso_x + 300), int(iso_y + 300)

# Crear lienzo en blanco
canvas = np.ones((600, 600, 3), dtype=np.uint8) * 255

# --- Ejes ---
cv2.line(canvas, iso_transform(0,0,0), iso_transform(200,0,0), (0,0,0), 2)  # eje X
cv2.putText(canvas, "X", iso_transform(200,0,0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

cv2.line(canvas, iso_transform(0,0,0), iso_transform(0,200,0), (0,0,0), 2)  # eje Y
cv2.putText(canvas, "Y", iso_transform(0,200,0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

cv2.line(canvas, iso_transform(0,0,0), iso_transform(0,0,150), (0,0,0), 2)  # eje Z
cv2.putText(canvas, "Z", iso_transform(0,0,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

# --- Prisma base ---
base_points = [iso_transform(0,0,0), iso_transform(100,0,0),
               iso_transform(100,60,0), iso_transform(0,60,0)]
cv2.polylines(canvas, [np.array(base_points)], True, (0,0,0), 2)

# --- Prisma superior ---
top_points = [iso_transform(0,0,50), iso_transform(100,0,50),
              iso_transform(100,60,50), iso_transform(0,60,50)]
cv2.polylines(canvas, [np.array(top_points)], True, (0,0,0), 2)

# --- Conexión vertical ---
for i in range(4):
    cv2.line(canvas, base_points[i], top_points[i], (0,0,0), 2)

# --- Superficie inclinada (ejemplo: conecta parte superior con otro nivel) ---
slope_points = [iso_transform(100,0,50), iso_transform(150,30,0),
                iso_transform(150,90,0), iso_transform(100,60,50)]
cv2.polylines(canvas, [np.array(slope_points)], True, (0,0,0), 2)

# --- Líneas ocultas (discontinuas) ---
cv2.line(canvas, iso_transform(0,60,0), iso_transform(0,60,50), (0,0,0), 1, cv2.LINE_AA)
cv2.line(canvas, iso_transform(100,60,0), iso_transform(100,60,50), (0,0,0), 1, cv2.LINE_AA)

# Mostrar resultado
cv2.imshow("Figura Isometrica 3D", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()