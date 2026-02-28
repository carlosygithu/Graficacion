import numpy as np
import cv2 as cv

# Crear matriz 10x10 con ceros (negro)
img = np.zeros((10,10), dtype=np.uint8)

# Dibujar pixel art (valores 255 = blanco)
img[2,2] = 255
img[2,7] = 255
img[5,4] = 255
img[7:,7] = 255

# Agrandar imagen
img_grande = cv.resize(img, (300,300), interpolation=cv.INTER_NEAREST)

cv.imshow("Pixel Art", img_grande)
cv.waitKey(0)
cv.destroyAllWindows()