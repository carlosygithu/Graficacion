import cv2 
import numpy as np

img = cv2.imread("C:\\Users\\Asus\\OneDrive\\Documentos\\GRAFICACION\\TareasGrafi\\frutas.png")


if img is None:
    print("Error al cargar la imagen")
    exit()

# Convertir a HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Rango para frutas rojas
lower_red = np.array([0,120,70])
upper_red = np.array([10,255,255])

# Crear máscara
mask = cv2.inRange(hsv, lower_red, upper_red)

# Kernel para limpieza
kernel = np.ones((5,5), np.uint8)

# Limpiar ruido
mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Detectar componentes conectados
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_limpia)

contador = 0

print("Analisis de regiones detectadas")
print("--------------------------------")

# Analizar regiones
for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    # Filtrar regiones pequeñas
    if area > 500:
        contador += 1
        print("Region", contador)
        print("Area aproximada:", area)
        print("----------------")

print("Numero total de frutas detectadas:", contador)

# Mostrar únicamente la máscara
cv2.imshow("Mascara Analizada", mask_limpia)

cv2.waitKey(0)
cv2.destroyAllWindows()

