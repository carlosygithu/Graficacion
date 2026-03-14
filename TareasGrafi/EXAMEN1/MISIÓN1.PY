import cv2 
import numpy as np
# Crear imagen en escala de grises
img = cv2.imread('C:\\Users\\Asus\\OneDrive\\Documentos\\GRAFICACION\\TareasGrafi\\EXAMEN1\\m1_oscura.png', cv2.IMREAD_GRAYSCALE)

# Verificar que la imagen se ha cargado correctamente
if img is None:
    print("Error al cargar la imagen")
    exit()

# MODO RAW: Multiplicar por *50
recuperada_raw= np.zeros_like(img)
# Recorremos cada pixel
for i in range(img.shape[0]):                    #filas
    for j in range(img.shape[1]): 
        valor= img[i,j]*50               #columnas
        recuperada_raw[i,j] = np.clip(valor,0,255)

# Guardar resultado
cv2.imwrite('recuperada_raw.png', recuperada_raw)

# MODO VECTORIZADO (Numpy / OpenCV)
recuperada_numpy= np.clip(img * 50, 0, 255).astype(np.uint8)
recuperada_cv = cv2.multiply(img, 50)

# Guardar resultado vectorizados
cv2.imwrite("recuperada_numpy.png", recuperada_numpy)
cv2.imshow("Recuperada RAW", recuperada_raw)

# Mostrar resultados 
cv2.imshow("Original Oscura", img)
cv2.imshow("Recuperada RAW", recuperada_raw)
cv2.imshow("Recuperada Numpy", recuperada_numpy)
cv2.imshow("Recuperada OpenCV", recuperada_cv)

cv2.waitKey(0)
cv2.destroyAllWindows()