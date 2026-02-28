import cv2 as cv
import numpy as np


# Crea una imagen negra de 600x600 píxeles
img = np.zeros((600,600), dtype=np.uint8)

# Define el centro de la imagen
centro_x = 300
centro_y = 300

# Genera valores de t desde 0 hasta 2π
# 1000 puntos para mayor precisión
t = np.linspace(0, 2*np.pi, 1000)


# ECUACIÓN PARAMÉTRICA 1: CÍRCULO
# x = r cos(t)
# y = r sin(t)
radio = 100

for i in range(len(t)):

    x = int(centro_x + radio*np.cos(t[i]))
    y = int(centro_y + radio*np.sin(t[i]))

    img[y,x] = 255


# ECUACIÓN PARAMÉTRICA 2: ELIPSE
# x = a cos(t)
# y = b sin(t)
a = 150
b = 50

for i in range(len(t)):

    x = int(centro_x + a*np.cos(t[i]))
    y = int(centro_y + b*np.sin(t[i]))

    img[y,x] = 200


# ECUACIÓN PARAMÉTRICA 3: ESPIRAL
# x = t cos(t)
# y = t sin(t)
for i in range(len(t)):

    x = int(centro_x + (t[i]*50)*np.cos(t[i]))
    y = int(centro_y + (t[i]*50)*np.sin(t[i]))

    if 0 <= x < 600 and 0 <= y < 600:
        img[y,x] = 150


# ECUACIÓN PARAMÉTRICA 4: ROSA
# x = r cos(4t) cos(t)
# y = r cos(4t) sin(t)
for i in range(len(t)):

    r = 100*np.cos(4*t[i])

    x = int(centro_x + r*np.cos(t[i]))
    y = int(centro_y + r*np.sin(t[i]))

    if 0 <= x < 600 and 0 <= y < 600:
        img[y,x] = 255


# ECUACIÓN PARAMÉTRICA 5: LEMNISCATA
# x = a cos(t) / (1 + sin²(t))
# y = a sin(t) cos(t) / (1 + sin²(t))
a = 200

for i in range(len(t)):

    x = int(centro_x + (a*np.cos(t[i]))/(1+np.sin(t[i])**2))
    y = int(centro_y + (a*np.sin(t[i])*np.cos(t[i]))/(1+np.sin(t[i])**2))

    if 0 <= x < 600 and 0 <= y < 600:
        img[y,x] = 255


# Muestra la imagen
cv.imshow("Ecuaciones Parametricas", img)

# Espera tecla
cv.waitKey(0)

# Cierra ventanas
cv.destroyAllWindows()