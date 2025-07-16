import os
import numpy as np
import cv2

def calcular_psnr(imagen_original, imagen_procesada):
    if imagen_original.shape != imagen_procesada.shape:
        raise ValueError("Las imágenes deben tener el mismo tamaño para comparar")
    mse = np.mean((imagen_original.astype(np.float32) - imagen_procesada.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def evaluar_carpeta(input_folder_original, input_folder_resultados):
    """
    Compara imágenes procesadas vs las imágenes con ruido (no ground truths),
    para calcular el PSNR.

    Parámetros:
        input_folder_original (str): carpeta con imágenes con ruido (bajo, medio, alto)
        input_folder_resultados (str): carpeta con imágenes procesadas por algoritmo

    Retorna:
        dict: PSNR por imagen
        float: PSNR promedio
    """
    psnr_resultados = {}
    suma = 0
    cantidad = 0

    for nombre_img in os.listdir(input_folder_resultados):
        if not nombre_img.endswith(".png"):
            continue

        # Buscar la imagen con ruido (nombre exacto igual)
        path_ori = os.path.join(input_folder_original, nombre_img)
        path_res = os.path.join(input_folder_resultados, nombre_img)

        if not (os.path.exists(path_ori) and os.path.exists(path_res)):
            print(f"❌ No encontrado (original o resultado): {nombre_img}")
            continue

        img_ori = cv2.imread(path_ori, cv2.IMREAD_GRAYSCALE)
        img_res = cv2.imread(path_res, cv2.IMREAD_GRAYSCALE)

        if img_ori is None:
            print(f"❌ No se pudo leer original: {path_ori}")
            continue
        if img_res is None:
            print(f"❌ No se pudo leer resultado: {path_res}")
            continue

        try:
            psnr = calcular_psnr(img_ori, img_res)
            psnr_resultados[nombre_img] = round(psnr, 2)
            suma += psnr
            cantidad += 1
        except Exception as e:
            print(f"❌ Error procesando {nombre_img}: {e}")

    promedio = suma / cantidad if cantidad > 0 else 0
    return psnr_resultados, promedio
