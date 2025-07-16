import os
import cv2
import csv
import time

from utils.ruido import agregar_ruido_gaussiano
from utils.bordes import aplicar_algoritmos_a_folder
from utils.evaluacion import evaluar_carpeta
from utils.evaluacion_f1 import evaluar_f1_batch

# Rutas
ruta_original_bsds = "imagenes/gt/originales_bsds"
ruta_con_ruido = "imagenes/con_ruido"
ruta_resultados = "imagenes/resultados"
ruta_gt = "imagenes/gt/test_png"
ruta_csv = "datos/resultados.csv"

# Niveles de ruido
niveles = {
    "bajo": 10,
    "medio": 25,
    "alto": 50
}

algoritmos = ["sobel", "prewitt", "canny"]

def obtener_nombres_validos():
    """Extrae nombres base (sin extensión) de las imágenes que tienen GT .png"""
    return [f.replace(".png", "") for f in os.listdir(ruta_gt) if f.endswith(".png")]

def procesar_imagenes_con_ruido():
    nombres_validos = obtener_nombres_validos()
    for nombre_base in nombres_validos:
        nombre_img = f"{nombre_base}.jpg"
        path_img = os.path.join(ruta_original_bsds, nombre_img)
        imagen = cv2.imread(path_img)

        if imagen is None:
            print(f"⚠️ No se pudo leer: {nombre_img}")
            continue

        for nivel, varianza in niveles.items():
            imagen_ruido = agregar_ruido_gaussiano(imagen, varianza)
            carpeta_nivel = os.path.join(ruta_con_ruido, nivel)
            os.makedirs(carpeta_nivel, exist_ok=True)

            salida = os.path.join(carpeta_nivel, f"{nombre_base}_{nivel}.png")
            cv2.imwrite(salida, imagen_ruido)
            print(f"✅ Guardado: {salida}")

def aplicar_algoritmos():
    for nivel in niveles.keys():
        carpeta_entrada = os.path.join(ruta_con_ruido, nivel)
        aplicar_algoritmos_a_folder(carpeta_entrada, ruta_resultados)

def evaluar_y_guardar():
    os.makedirs("datos", exist_ok=True)
    with open(ruta_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Imagen", "Nivel_Ruido", "Algoritmo", "PSNR", "F1-Score", "Tiempo_ms"])

        nombres_validos = obtener_nombres_validos()

        for nivel in niveles.keys():
            carpeta_con_ruido = os.path.join(ruta_con_ruido, nivel)
            sufijo = f"_{nivel}.png"

            for algoritmo in algoritmos:
                carpeta_resultado = os.path.join(ruta_resultados, algoritmo)

                # ✅ PSNR: comparar (resultado, con_ruido)
                psnrs, _ = evaluar_carpeta(carpeta_resultado, carpeta_con_ruido)
                f1s, _ = evaluar_f1_batch(ruta_gt, carpeta_resultado, sufijo=sufijo)

                for nombre in nombres_validos:
                    nombre_img = f"{nombre}{sufijo}"
                    path_img = os.path.join(carpeta_resultado, nombre_img)

                    if not os.path.exists(path_img):
                        print(f"❌ No encontrado (original o resultado): {nombre_img}")
                        continue

                    # Medir tiempo de lectura
                    inicio = time.time()
                    _ = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
                    fin = time.time()
                    tiempo_ms = round((fin - inicio) * 1000, 2)

                    psnr = round(psnrs.get(nombre_img, 0), 2)
                    f1 = round(f1s.get(nombre_img, 0), 4)

                    writer.writerow([nombre_img, nivel, algoritmo, psnr, f1, tiempo_ms])

                print(f"📊 Resultados para {algoritmo.upper()} ({nivel}) listos.")

if __name__ == "__main__":
    print("🔄 Generando imágenes con ruido (solo dataset BSDS500 en .png)...")
    procesar_imagenes_con_ruido()

    print("\n⚙️ Aplicando algoritmos de detección de bordes...")
    aplicar_algoritmos()

    print("\n📈 Evaluando PSNR, F1-score y tiempo de ejecución...")
    evaluar_y_guardar()

    print("\n✅ Todo el proceso se completó correctamente.")
