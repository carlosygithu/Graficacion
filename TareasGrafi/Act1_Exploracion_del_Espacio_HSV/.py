import cv2
import numpy as np

# cargar imagen
img = cv2.imread("C:\\Users\\Asus\\OneDrive\\Documentos\\GRAFICACION\\TareasGrafi\\frutas.png")

# convertir a HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# rango HSV para color rojo
lower_red = np.array([0,120,70])
upper_red = np.array([10,255,255])

# crear mascara para color rojo
mask_red = cv2.inRange(hsv, lower_red, upper_red)

# mostrar resultados
cv2.imshow("Imagen Original", img)
cv2.imshow("Imagen HSV", hsv)
cv2.imshow("Mascara Roja", mask_red)

# Esperar hasta que se presione una tecla
cv2.waitKey(0)
# Cerar ventanas
cv2.destroyAllWindows()
