# Evaluación de Algoritmos de Detección de Bordes con Ruido Gaussiano

Este proyecto tiene como objetivo comparar distintos algoritmos de detección de bordes (clásicos y de deep learning) aplicados sobre imágenes con diferentes niveles de ruido. Se utilizan métricas cuantitativas como PSNR, F1-Score y tiempo de ejecución para medir el desempeño.

## 🧪 Algoritmos Evaluados

- Sobel
- Prewitt
- Canny
- PiDiNet
- DexiNed

## 📈 Métricas

- **PSNR**: calidad visual comparando con la imagen con ruido
- **F1-Score**: precisión comparando con la ground truth binarizada
- **Tiempo de ejecución** por imagen

---

## 🚀 Requisitos

1. Python ≥ 3.8
2. Instalar dependencias:

## Instalar librerias requeridas
pip install -r requirements.txt

## Contenido de requirements.txt

1. numpy
2. pencv-python
3. torch>=1.9.0
4. torchvision
5. matplotlib
6. scikit-learn
7. tqdm


