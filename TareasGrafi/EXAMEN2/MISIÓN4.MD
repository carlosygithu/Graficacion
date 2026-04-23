
# Misión 4: La Frecuencia Térmica II (HSV + máscara + limpieza por convolución)

---

## 1. Introducción
Se ocultó la contraseña en la imagen `m4_ruido.png`.  
Sin embargo, el ruido presente generaba falsos positivos al segmentar el color Cyan.  
El objetivo fue aplicar un filtro de suavizado y luego segmentar en HSV para obtener una máscara más limpia que revelara la contraseña.

---

## 2. Objetivo
    * Reducir el ruido mediante un filtro de convolución promedio 3x3.  
    * Convertir la imagen al espacio de color HSV.  
    * Segmentar el color Cyan (Hue ~ 90).  
    * Generar y guardar la máscara resultante. 

---

# 3. Codigo
```python

import cv2
import numpy as np

# Cargar la imagen con ruido
img = cv2.imread("m4_ruido.png")

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

```

---

# 4. Resultados
m4_suavizada.png: imagen suavizada con menos ruido, donde el texto Cyan se distingue mejor.

m4_mask_cyan.png: máscara binaria que revela claramente la contraseña oculta.

La contraseña encontrada es:
![alt text](image-4.png)
![alt text](image-5.png)
---

# 5. Análisis
   * El filtro promedio redujo el ruido aleatorio, mejorando la visibilidad del texto.

   * La conversión a HSV permitió aislar el color Cyan de manera robusta.

   * La máscara final eliminó falsos positivos y reveló la contraseña de forma precisa.

---

# 6. Conclusión
Se logró limpiar la imagen y segmentar correctamente el color Cyan.
La contraseña oculta fue revelada exitosamente, cumpliendo los objetivos de la misión.