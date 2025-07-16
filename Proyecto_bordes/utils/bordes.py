import os
import cv2
import numpy as np

def aplicar_sobel(imagen):
    sobelx = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    return np.clip(sobel, 0, 255).astype(np.uint8)

def aplicar_prewitt(imagen):
    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])
    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]])
    prewittx = cv2.filter2D(imagen, -1, kernelx)
    prewitty = cv2.filter2D(imagen, -1, kernely)
    prewitt = cv2.add(prewittx, prewitty)
    return prewitt

def aplicar_canny(imagen):
    return cv2.Canny(imagen, 100, 200)

def aplicar_algoritmos_a_folder(input_folder, output_base):
    if not os.path.exists(input_folder):
        print(f"⚠️ Carpeta no encontrada: {input_folder}")
        return

    for nombre_img in os.listdir(input_folder):
        path = os.path.join(input_folder, nombre_img)
        imagen = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            print(f"⚠️ Imagen no válida: {nombre_img}")
            continue

        # Aplicar algoritmos
        resultados = {
            "sobel": aplicar_sobel(imagen),
            "prewitt": aplicar_prewitt(imagen),
            "canny": aplicar_canny(imagen)
        }

        # Guardar resultados
        for nombre_algo, img_resultado in resultados.items():
            out_dir = os.path.join(output_base, nombre_algo)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, nombre_img)
            cv2.imwrite(out_path, img_resultado)
            print(f"✅ Guardado {nombre_algo}: {out_path}")
