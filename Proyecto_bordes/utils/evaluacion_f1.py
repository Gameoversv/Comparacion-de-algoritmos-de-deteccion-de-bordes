import os
import cv2
import numpy as np
from sklearn.metrics import f1_score

def calcular_f1(gt_path, pred_path):
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if gt is None or pred is None:
        return None

    # Normalizar (convertir a 0 y 1)
    gt_bin = (gt > 127).astype(np.uint8).flatten()
    pred_bin = (pred > 127).astype(np.uint8).flatten()

    return f1_score(gt_bin, pred_bin)

def evaluar_f1_batch(ruta_gt_png, ruta_resultados, sufijo=""):
    """
    Evalúa F1-score para imágenes en ruta_resultados, usando ground truths en ruta_gt_png.
    Usa sufijo como "_bajo.png", "_medio.png", etc., para hacer match.

    Parámetros:
        ruta_gt_png (str): Carpeta con ground truths (.png)
        ruta_resultados (str): Carpeta con resultados generados
        sufijo (str): Sufijo que debe coincidir al final del nombre, ej: "_bajo.png"

    Retorna:
        dict: f1 por archivo
        float: f1 promedio
    """
    f1_scores = {}
    suma = 0
    cantidad = 0

    for nombre in os.listdir(ruta_resultados):
        if not nombre.endswith(sufijo):
            continue

        # Extraer el nombre base
        nombre_base = nombre.replace(sufijo, ".png")
        path_gt = os.path.join(ruta_gt_png, nombre_base)
        path_pred = os.path.join(ruta_resultados, nombre)

        if not os.path.exists(path_gt):
            print(f"❌ Ground truth no encontrado: {path_gt}")
            continue

        f1 = calcular_f1(path_gt, path_pred)
        if f1 is not None:
            f1_scores[nombre] = round(f1, 4)
            suma += f1
            cantidad += 1

    promedio = suma / cantidad if cantidad > 0 else 0
    return f1_scores, promedio
