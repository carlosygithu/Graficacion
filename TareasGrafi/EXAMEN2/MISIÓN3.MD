# Misión 3: El Sello Biométrico II (Primitivas + Simetría)

---

## 1. Introducción
El servidor principal requiere un sello geométrico que combine primitivas gráficas, simetría y una marca de autenticidad.  
El objetivo es construir un diseño exacto siguiendo las especificaciones dadas.

---

## 2. Objetivo
    * Crear un lienzo 600x600 con color base BGR(40,20,20).  
    * Dibujar círculos concéntricos amarillos.  
    * Insertar un rectángulo rojo sólido en el centro.  
    * Añadir dos diagonales blancas formando una “X”.  
    * Colocar 8 círculos verdes simétricos alrededor del centro.  
    * Escribir el texto `SECTOR-9` en la parte inferior.  
    * Guardar la evidencia como `m3_sello_forjado_v2.png`.

---

# 3. Codigo
```python

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

```

---

# 4. Resultados
El archivo generado m3_sello_forjado_v2.png muestra el sello geométrico con todos los elementos solicitados:

![alt text](image-3.png)
---

# 5. Análisis
    * El uso de primitivas gráficas permite construir figuras precisas y reproducibles.

    * La simetría de los círculos verdes asegura autenticidad visual.

    * El orden de dibujo garantiza que cada elemento se superponga correctamente.
---

# 6. Conclusión
Se logró construir el sello biométrico con simetría y autenticidad, cumpliendo las especificaciones exactas de la misión.
El resultado final es un diseño geométrico completo y verificable.