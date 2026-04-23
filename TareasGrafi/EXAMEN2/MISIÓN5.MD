
# Misión 5: La Huella de Canales (Separación BGR + combinación)

---

## 1. Introducción
Se interceptó una imagen donde la clave no estaba en HSV, sino en una diferencia entre canales BGR.  
El objetivo fue generar una imagen con ruido, ocultar un mensaje en un canal específico y luego recuperarlo mediante separación y combinación de canales.

---

## 2. Objetivo
    * Generar una imagen de 300x700 con ruido aleatorio en BGR.  
    * Escribir un texto con “tinta tramposa” dependiente de un canal.  
    * Guardar la evidencia como `m5_tricolor.png`.  
    * Recuperar el mensaje probando canales y combinaciones (`abs(G - B)`).  
    * Guardar la mejor recuperación como `m5_mensaje.png`.  

---

# 3. Codigo
```python

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
```

---

# 4. Resultados
   * m5_tricolor.png: imagen con ruido y texto oculto en canales.

   * m5_mensaje.png: recuperación clara del mensaje oculto mediante la combinación abs(G - B).

El mensaje revelado es:
![alt text](image-6.png)
---

# 5. Análisis
* La separación de canales permite   observar diferencias de intensidad.

* La combinación abs(G - B) reveló el mensaje con mayor claridad.

*  El uso de normalización mejoró el  contraste y la legibilidad del texto.
---

# 6. Conclusión
Se logró ocultar y recuperar un mensaje utilizando diferencias entre canales BGR.
La técnica demostró cómo la manipulación de canales puede servir para esconder y revelar información en imágenes digitales.