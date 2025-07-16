import numpy as np
import cv2

def agregar_ruido_gaussiano(imagen, varianza=20):
    """
    Convierte la imagen a escala de grises si es necesario,
    y le agrega ruido gaussiano con la varianza especificada.

    Parámetros:
        imagen (ndarray): Imagen original (BGR o escala de grises)
        varianza (float): Varianza del ruido gaussiano

    Retorna:
        ndarray: Imagen en escala de grises con ruido
    """
    if imagen is None:
        raise ValueError("La imagen es None")

    # Si la imagen tiene 3 canales, se convierte a escala de grises
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    fila, col = imagen.shape
    ruido = np.random.normal(0, varianza ** 0.5, (fila, col))
    imagen_ruido = imagen.astype(np.float32) + ruido

    return np.clip(imagen_ruido, 0, 255).astype(np.uint8)
