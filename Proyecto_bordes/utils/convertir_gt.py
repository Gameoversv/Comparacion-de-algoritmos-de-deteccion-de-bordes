import os
import scipy.io
import numpy as np
import cv2

# Rutas de entrada/salida
ruta_mats = "../imagenes/gt/test"
ruta_salida = "../imagenes/gt/test_png"

os.makedirs(ruta_salida, exist_ok=True)

# Función para extraer y guardar la primera máscara
def convertir_mascara(mat_path, salida_path):
    data = scipy.io.loadmat(mat_path)
    gt_list = data['groundTruth'][0]

    if len(gt_list) == 0:
        print(f"❌ No se encontró ground truth en {mat_path}")
        return

    # Seleccionamos la primera anotación humana
    gt_1 = gt_list[0]
    mask = gt_1['Boundaries'][0, 0]  # matriz booleana

    # Convertimos a formato binario (0-255)
    mask_binaria = (mask * 255).astype(np.uint8)

    # Guardar como imagen PNG
    cv2.imwrite(salida_path, mask_binaria)
    print(f"✅ Ground truth exportado: {salida_path}")

# Procesar algunos archivos como ejemplo
archivos = os.listdir(ruta_mats)
ejemplo = [f for f in archivos if f.endswith('.mat')][:15]  # primeros 10

for nombre in ejemplo:
    entrada = os.path.join(ruta_mats, nombre)
    nombre_salida = nombre.replace(".mat", ".png")
    salida = os.path.join(ruta_salida, nombre_salida)
    convertir_mascara(entrada, salida)
