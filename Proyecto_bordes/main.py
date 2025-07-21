import os
import cv2
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt

from utils.ruido import agregar_ruido_gaussiano
from utils.bordes import aplicar_algoritmos_a_folder
from utils.evaluacion import evaluar_carpeta
from utils.evaluacion_f1 import evaluar_f1_batch

# === Configuración ===
REPETICIONES = 5
ruta_original_bsds = "imagenes/gt/originales_bsds"
ruta_con_ruido_base = "imagenes/con_ruido/base"
ruta_resultados = "imagenes/resultados"
ruta_gt = "imagenes/gt/test_png"
carpeta_csvs = "datos"
os.makedirs(carpeta_csvs, exist_ok=True)

niveles = {
    "bajo": 10,
    "medio": 25,
    "alto": 50
}
algoritmos = ["sobel", "prewitt", "canny", "pidinet", "dexined"]

def obtener_nombres_validos():
    return [f.replace(".png", "") for f in os.listdir(ruta_gt) if f.endswith(".png")]

def generar_ruido_base():
    print("🧪 Generando imágenes con ruido base (una vez)...")
    nombres_validos = obtener_nombres_validos()
    for nombre_base in nombres_validos:
        nombre_img = f"{nombre_base}.jpg"
        path_img = os.path.join(ruta_original_bsds, nombre_img)
        imagen = cv2.imread(path_img)
        if imagen is None:
            print(f"⚠️ Imagen no encontrada o inválida: {nombre_img}")
            continue
        for nivel, varianza in niveles.items():
            carpeta_nivel = os.path.join(ruta_con_ruido_base, nivel)
            os.makedirs(carpeta_nivel, exist_ok=True)
            salida = os.path.join(carpeta_nivel, f"{nombre_base}_{nivel}.png")
            if not os.path.exists(salida):
                ruido = agregar_ruido_gaussiano(imagen, varianza)
                cv2.imwrite(salida, ruido)
                print(f"✅ Ruido {nivel} guardado: {salida}")

def aplicar_algoritmos(ronda):
    out_base = os.path.join(ruta_resultados, ronda)
    tiempos_ronda = {}

    for nivel in niveles:
        carpeta_entrada = os.path.join(ruta_con_ruido_base, nivel)
        print(f"📂 Aplicando algoritmos en nivel '{nivel}'...")
        tiempos_nivel = aplicar_algoritmos_a_folder(carpeta_entrada, out_base)
        if not tiempos_nivel:
            print(f"⚠️ No se registraron tiempos para nivel '{nivel}'.")

        for img, tiempos_algos in tiempos_nivel.items():
            tiempos_ronda[f"{img}|{nivel}"] = tiempos_algos

    return tiempos_ronda

def evaluar(ronda, tiempos_ronda):
    archivo = os.path.join(carpeta_csvs, f"resultados_{ronda}.csv")
    nombres_validos = obtener_nombres_validos()
    with open(archivo, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Imagen", "Nivel_Ruido", "Algoritmo", "PSNR", "F1-Score", "Tiempo_ms"])

        for nivel in niveles:
            sufijo = f"_{nivel}.png"
            carpeta_ruido = os.path.join(ruta_con_ruido_base, nivel)
            for algoritmo in algoritmos:
                carpeta_resultado = os.path.join(ruta_resultados, ronda, algoritmo)
                psnrs, _ = evaluar_carpeta(carpeta_resultado, carpeta_ruido)
                f1s, _ = evaluar_f1_batch(ruta_gt, carpeta_resultado, sufijo=sufijo)

                for nombre in nombres_validos:
                    imagen = f"{nombre}{sufijo}"
                    path_img = os.path.join(carpeta_resultado, imagen)
                    if not os.path.exists(path_img):
                        print(f"⏭️ Saltando (no existe resultado): {imagen}")
                        continue

                    psnr = round(psnrs.get(imagen, 0), 2)
                    f1 = round(f1s.get(imagen, 0), 4)
                    tiempo_ms = tiempos_ronda.get(f"{imagen}|{nivel}", {}).get(algoritmo, 0.0)
                    writer.writerow([imagen, nivel, algoritmo, psnr, f1, tiempo_ms])

def calcular_promedios_y_graficar():
    csvs = [os.path.join(carpeta_csvs, f"resultados_ronda{i+1}.csv") for i in range(REPETICIONES)]
    dfs = [pd.read_csv(f) for f in csvs if os.path.exists(f)]
    if not dfs:
        print("⚠️ No hay resultados CSV para calcular promedios.")
        return

    df_total = pd.concat(dfs, ignore_index=True)
    promedio = df_total.groupby(["Algoritmo", "Nivel_Ruido"]).agg({
        "PSNR": "mean",
        "F1-Score": "mean",
        "Tiempo_ms": "mean"
    }).reset_index()
    promedio.to_csv(os.path.join(carpeta_csvs, "resultados_promedios.csv"), index=False)

    for metrica in ["PSNR", "F1-Score", "Tiempo_ms"]:
        plt.figure(figsize=(10, 5))
        for nivel in promedio["Nivel_Ruido"].unique():
            sub = promedio[promedio["Nivel_Ruido"] == nivel]
            plt.plot(sub["Algoritmo"], sub[metrica], marker='o', label=nivel)
        plt.title(f"{metrica} promedio por algoritmo y nivel de ruido")
        plt.xlabel("Algoritmo")
        plt.ylabel(metrica)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_csvs, f"{metrica}_promedio.png"))
        plt.close()

# === Ejecución principal ===
if __name__ == "__main__":
    generar_ruido_base()

    for i in range(REPETICIONES):
        ronda = f"ronda{i+1}"
        print(f"\n🔁 Iniciando {ronda.upper()}...")
        tiempos = aplicar_algoritmos(ronda)
        evaluar(ronda, tiempos)
        print(f"✅ {ronda.upper()} completada.")

    print("\n📊 Calculando promedios y generando gráficas...")
    calcular_promedios_y_graficar()
    print("🎉 Proceso completo.")
