import cv2
import numpy as np

img = cv2.imread("C:\\Users\\Asus\\OneDrive\\Documentos\\GRAFICACION\\TareasGrafi\\frutas.png")

if img is None:
    print("Error al cargar la imagen")
    exit()

# Convertir imagen a HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)        

# Definir rango de color rojo
lower_red = np.array([0,120,70])
upper_red = np.array([10,255,255])

# Crear máscara binaria
mask = cv2.inRange(hsv, lower_red, upper_red)

# Crear kernel para operaciones morfológicas
kernel = np.ones((5,5), np.uint8)

# Aplicar operación de apertura para eliminar ruido
mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Mostrar resultados
cv2.imshow("Mascara Original", mask)
cv2.imshow("Mascara Limpia", mask_limpia)

cv2.imshow("Imagen Original", img)

cv2.waitKey(0)
cv2.destroyAllWindows()