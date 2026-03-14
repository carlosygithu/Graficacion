import cv2
import numpy as np
import math

# Crear lienzo negro de 500x500 
lienzo = np.zeros((500, 500, 3), dtype=np.uint8)

# Bucle para t de 0 a 2π con incrementos de 0.01
t = np.arange(0, 2*math.pi, 0.01)
for valor in t:
    x = int(250 + 150 * math.sin(3*valor))
    y = int(250 + 150 * math.sin(2*valor))
    # dibujar punto blanco en esas coordenadas
    cv2.circle(lienzo, (x, y), 1, (255, 255, 255), -1)

# Mostrar el lienzo
cv2.imshow("Cuerva de Lissajous - Antena", lienzo)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar en el disco
cv2.imwrite("m5_antena.png", lienzo)