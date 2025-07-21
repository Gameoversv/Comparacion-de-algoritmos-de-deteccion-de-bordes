import os
import cv2
import numpy as np
import torch
import types
import torch.nn.functional as F
import time


# === Importar modelos ===
from pidinet.models import pidinet
from DexiNed.model import DexiNed  # ✅ Correcto si está en Proyecto_bordes/DexiNed/model.py


# === Dispositivo (GPU si disponible) ===
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === PiDiNet ===
args = types.SimpleNamespace()
args.config = 'carv4'
args.dil = True
args.sa = True
_model_pidinet = pidinet(args).to(_device)

weights_path = os.path.join("pidinet", "trained_models", "table5_pidinet.pth")
checkpoint = torch.load(weights_path, map_location=_device)
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
_model_pidinet.load_state_dict(state_dict, strict=False)
_model_pidinet.eval()

# === DexiNed ===
_model_dexined = DexiNed().to(_device)
dexi_ckpt_path = os.path.join("DexiNed", "checkpoints", "BIPED", "10", "10_model.pth")
_model_dexined.load_state_dict(torch.load(dexi_ckpt_path, map_location=_device))
_model_dexined.eval()

# === Algoritmos clásicos ===

def aplicar_sobel(imagen):
    sobelx = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    return np.clip(sobel, 0, 255).astype(np.uint8)

def aplicar_prewitt(imagen):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittx = cv2.filter2D(imagen, -1, kernelx)
    prewitty = cv2.filter2D(imagen, -1, kernely)
    return cv2.add(prewittx, prewitty)

def aplicar_canny(imagen):
    return cv2.Canny(imagen, 100, 200)

# === PiDiNet ===

@torch.no_grad()
def aplicar_pidinet(imagen):
    h, w = imagen.shape[:2]
    img = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img, (320, 320))
    tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(_device)
    salida = _model_pidinet(tensor)[-1]
    salida = F.interpolate(salida, size=(h, w), mode='bilinear', align_corners=False)
    salida = salida.squeeze().cpu().numpy()
    return (salida * 255).astype(np.uint8)

# === DexiNed ===

@torch.no_grad()
def aplicar_dexined(imagen):
    h, w = imagen.shape[:2]
    img = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img, (512, 512))
    tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(_device)
    salida = _model_dexined(tensor)[-1]  # última salida
    salida = F.interpolate(salida, size=(h, w), mode='bilinear', align_corners=False)
    salida = salida.squeeze().cpu().numpy()
    return (salida * 255).astype(np.uint8)

# === Aplicar todos los algoritmos ===

def aplicar_algoritmos_a_folder(input_folder, output_base):
    if not os.path.exists(input_folder):
        print(f"⚠️ Carpeta no encontrada: {input_folder}")
        return {}

    tiempos_totales = {}

    for nombre_img in os.listdir(input_folder):
        path = os.path.join(input_folder, nombre_img)
        imagen = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            print(f"⚠️ Imagen no válida: {nombre_img}")
            continue

        resultados = {}
        tiempos = {}

        # Medir tiempos individualmente
        for nombre_algo, funcion in {
            "sobel": aplicar_sobel,
            "prewitt": aplicar_prewitt,
            "canny": aplicar_canny,
            "pidinet": aplicar_pidinet,
            "dexined": aplicar_dexined
        }.items():
            inicio = time.time()
            resultado = funcion(imagen)
            fin = time.time()
            duracion_ms = round((fin - inicio) * 1000, 2)
            resultados[nombre_algo] = resultado
            tiempos[nombre_algo] = duracion_ms

        # Guardar resultados
        for nombre_algo, img_resultado in resultados.items():
            out_dir = os.path.join(output_base, nombre_algo)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, nombre_img)
            cv2.imwrite(out_path, img_resultado)
            print(f"✅ Guardado {nombre_algo}: {out_path}")

        # Asociar los tiempos a la imagen
        tiempos_totales[nombre_img] = tiempos

    return tiempos_totales
