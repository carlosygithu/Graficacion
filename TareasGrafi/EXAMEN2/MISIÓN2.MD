
# Misión 2: El QR Fragmentado II (Traslación + Rotación + Ensamble)

---

## 1. Introducción
Se recibió un código QR dividido en dos mitades:  
  * La mitad superior desplazada.  
  * La mitad inferior rotada 180°.  

El objetivo es reconstruir el QR aplicando transformaciones geométricas inversas y ensamblando ambas piezas en un lienzo blanco.

---

## 2. Objetivo
- Crear un lienzo blanco de 400x400 píxeles.  
- Corregir la traslación de la mitad superior.  
- Corregir la rotación de la mitad inferior.  
- Ensamblar ambas piezas para obtener el QR completo.  
---

# 3. Codigo
```python

import cv2
import numpy as np

# Cargar las mitades del QR
mitad1 = cv2.imread('D:\\Proyectos\\GRAFICACION\\TareasGrafi\\EXAMEN2\\m2_mitad1 1.png')
mitad2 = cv2.imread('D:\\Proyectos\\GRAFICACION\\TareasGrafi\\EXAMEN2\\m2_mitad2 1.png')

# Crear lienzo blanco de 400x400
lienzo = np.full((400, 400, 3), 255, dtype=np.uint8)

# -------------------------------
# Corrección de la mitad superior
# -------------------------------
# Traslación inversa (ajusta dx, dy según el desplazamiento original)
# Aquí supongamos que estaba corrida 50 px a la derecha y 20 px abajo
M_traslacion = np.float32([[1, 0, -50],
                           [0, 1, -20]])
mitad1_corr = cv2.warpAffine(mitad1, M_traslacion, (400, 200))

# Pegar mitad superior en lienzo
lienzo[0:200, 0:400] = mitad1_corr

# -------------------------------
# Corrección de la mitad inferior
# -------------------------------
# Rotación inversa 180° sobre el centro de la imagen
(h, w) = mitad2.shape[:2]
centro = (w // 2, h // 2)
M_rotacion = cv2.getRotationMatrix2D(centro, 180, 1.0)
mitad2_corr = cv2.warpAffine(mitad2, M_rotacion, (w, h))

# Pegar mitad inferior en lienzo
lienzo[200:400, 0:400] = mitad2_corr

# -------------------------------
# Guardar resultado final
# -------------------------------
cv2.imwrite("m2_qr_reconstruido.png", lienzo)

```

---

# 4. Resultados
El archivo generado m2_qr_reconstruido.png muestra el QR reconstruido correctamente, con las dos mitades alineadas en un lienzo blanco de 400x400.

![alt text](image-2.png)
---

# 5. Análisis
     * La traslación inversa devuelve la mitad superior a su posición original.

     * La rotación inversa de 180° corrige la orientación de la mitad inferior.

     * El ensamblaje en el lienzo blanco permite visualizar el QR completo sin solapamientos.
     
---

# 6. Conclusión
Se logró reconstruir el QR fragmentado aplicando transformaciones geométricas inversas. El resultado final es un código QR completo y legible, cumpliendo los objetivos de la misión.