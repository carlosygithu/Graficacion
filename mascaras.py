import cv2 as cv
import numpy as np

# Inicia la cámara (0 = cámara principal)
cap = cv.VideoCapture(0)

while True:

    # Lee frame de la cámara
    ret, frame = cap.read()

    if not ret:
        break

    # Convierte a escala de grises
    gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Crea máscara con umbral
    # Todo lo mayor a 100 = blanco
    # Todo lo menor = negro
    _, mascara = cv.threshold(gris, 80, 255, cv.THRESH_BINARY)

    # Aplica máscara
    resultado = cv.bitwise_and(frame, frame, mask=mascara)

    # Muestra ventanas
    cv.imshow("Video original", frame)
    cv.imshow("Blanco y negro", gris)
    cv.imshow("Mascara aplicada", resultado)

    # Salir con tecla ESC
    if cv.waitKey(1) & 0xFF == 27:
        break


# Libera cámara
cap.release()

# Cierra ventanas
cv.destroyAllWindows()