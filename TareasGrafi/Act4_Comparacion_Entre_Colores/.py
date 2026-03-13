import cv2 
import numpy as np

# Cargar imagen
img = cv2.imread("C:\\Users\\Asus\\OneDrive\\Documentos\\GRAFICACION\\TareasGrafi\\frutas.png")
if img is None:
    print("Error al cargar la imagen")
    exit()

# Convertir a HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Definir rangos HSV para cada color
colores = {
"Rojo": ([0,120,70], [10,255,255]),
"Verde": ([35,80,80], [85,255,255]),
"Amarillo": ([20,100,100], [30,255,255])
}

# Kernel para limpieza
kernel = np.ones((5,5), np.uint8)
print("Analisis por color")
print("---------------------------")
for nombre,(lower,upper) in colores.items():
    lower = np.array(lower)
    upper = np.array(upper)

    # Crear máscara
    mask = cv2.inRange(hsv, lower, upper)

    # Limpiar ruido
    mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Detectar regiones conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_limpia)
    contador = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Filtrar regiones pequeñas
        if area > 500:
            contador += 1

    print("Color:", nombre)
    print("Frutas detectadas:", contador)
    print("---------------------------")
    cv2.imshow("Mascara " + nombre, mask_limpia)

cv2.imshow("Imagen Original", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
