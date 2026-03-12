import cv2
import numpy as np

# cargar imagen
img = cv2.imread("C:\\Users\\Asus\\OneDrive\\Documentos\\GRAFICACION\\TareasGrafi\\frutas.png")

# convertir a HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# rango para color rojo
lower_red = np.array([0,120,70])
upper_red = np.array([10,255,255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)

cv2.imshow("Imagen Original", img)
cv2.imshow("Imagen HSV", hsv)
cv2.imshow("Mascara Roja", mask_red)

cv2.waitKey(0)
cv2.destroyAllWindows()
