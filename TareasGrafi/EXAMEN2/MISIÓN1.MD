
# Misión 1: El Mensaje Subexpuesto II
---

# 1. Introducción
Se recibió una imagen oscura (`m1_oscura.png`) en la cual el mensaje fue ocultado mediante operaciones puntuales que redujeron su intensidad. El objetivo es aplicar transformaciones para recuperar el texto escondido.

---

# 2. Objetivo
Recuperar el mensaje oculto aplicando operadores puntuales en dos fases:  
- Multiplicación por 50.  
- Suma de +20.  
- Saturación en el rango 0..255.  

Generar dos evidencias:  
- `m1_recuperado_x50.png`  
- `m1_recuperado_x50_mas20.png`

---

# 3. Código
```python
import cv2
import numpy as np

# Cargar la imagen oscura
img = cv2.imread('D:\\Proyectos\\GRAFICACION\\TareasGrafi\\EXAMEN2\\m1_oscura 1.png',
 cv2.IMREAD_GRAYSCALE)

# -------------------------------
# MODO RAW (ciclos anidados)
# -------------------------------
recuperado_x50 = np.zeros_like(img, dtype=np.int32)

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        valor = img[y, x] * 50
        recuperado_x50[y, x] = np.clip(valor, 0, 255)

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
recuperado_numpy_x50 = np.clip(img * 50, 0, 255).astype(np.uint8)
cv2.imwrite("m1_recuperado_x50_numpy.png", recuperado_numpy_x50)

recuperado_numpy_x50_mas20 = np.clip(img * 50 + 20, 0, 255).astype(np.uint8)
cv2.imwrite("m1_recuperado_x50_mas20_numpy.png", recuperado_numpy_x50_mas20)

```

---

# 4. Resultados
Las imágenes recuperadas (m1_recuperado_x50.png y m1_recuperado_x50_mas20.png) revelan el mensaje oculto en tonos claros.
El vectorizado produce el mismo resultado pero con mayor eficiencia.

Los resultado del mensaje oculto son:
m1_recuperado_x50_mas20
![alt text](image.png)

m1_recuperado_x50_numpy
![alt text](image-1.png)

---

# 5. Análisis
   * El modo RAW permite comprender el proceso píxel por píxel.

   * El modo vectorizado con NumPy/OpenCV es más eficiente y produce el mismo resultado.

   * La saturación asegura que los valores se mantengan dentro del rango visible (0–255).

   * El mensaje oculto se recupera exitosamente aplicando las transformaciones puntuales.
# 6. Conclusión
Se logró recuperar el mensaje oculto ALBATROS_7 mediante operadores puntuales.
Ambos métodos (RAW y vectorizado) cumplen el objetivo, aunque el vectorizado es más práctico en aplicaciones reales.