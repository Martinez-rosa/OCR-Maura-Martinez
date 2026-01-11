# Guía de Ejecución y Requisitos del Sistema OCR

Este documento detalla cómo ejecutar el programa de reconocimiento óptico de caracteres (OCR) y las librerías necesarias para su funcionamiento.

## 🚀 Ejecución del Programa

El sistema cuenta con un punto de entrada principal `main.py` que permite dos modos de operación: Interfaz Gráfica (Web) y Línea de Comandos (CLI).

### 1. Interfaz Gráfica (Modo Recomendado)
Este modo inicia un servidor web local con una interfaz visual para cargar imágenes y ver resultados.

**Comando:**
```bash
python main.py --gui
```
Una vez iniciado, abra su navegador web y visite: `http://127.0.0.1:5000`

### 2. Línea de Comandos (CLI)
Permite procesar una imagen específica directamente desde la terminal sin interfaz gráfica.

**Comando Básico:**
```bash
python main.py --image "ruta/a/tu/imagen.png"
```

**Opciones Adicionales:**
- `--extract-images`: Detecta y extrae figuras/imágenes dentro del documento.
- `--output_dir "mi_carpeta"`: Especifica una carpeta de salida personalizada (por defecto es `output`).
- `--debug`: Guarda imágenes intermedias de la segmentación para depuración.

**Ejemplo Completo:**
```bash
python main.py --image "documento.jpg" --extract-images --output_dir "resultados_ocr"
```

---

## 📦 Librerías Externas Requeridas

Para el correcto funcionamiento del sistema, se requieren las siguientes librerías. Se recomienda utilizar **Python 3.8** o superior.

### Instalación Automática
Puede instalar todas las dependencias necesarias ejecutando:
```bash
pip install -r requirements.txt
```

### Detalle de Dependencias
A continuación se especifican las librerías, sus versiones requeridas y fuentes de descarga manual.

| Librería | Versión Mínima | Función Principal | Fuente Oficial | Comando de Instalación |
|----------|----------------|-------------------|----------------|------------------------|
| **OpenCV** (`opencv-python`) | 4.5.0 | Procesamiento de imágenes y visión artificial | [PyPI - opencv-python](https://pypi.org/project/opencv-python/) | `pip install opencv-python>=4.5.0` |
| **NumPy** | 1.20.0 | Operaciones matemáticas y manejo de matrices | [PyPI - numpy](https://pypi.org/project/numpy/) | `pip install numpy>=1.20.0` |
| **Flask** | (Reciente) | Servidor web para la interfaz gráfica | [PyPI - Flask](https://pypi.org/project/Flask/) | `pip install flask` |
| **TensorFlow** | (Reciente) | Ejecución de modelos de redes neuronales (CNN) | [PyPI - tensorflow](https://pypi.org/project/tensorflow/) | `pip install tensorflow` |
| **Pillow** | 9.0.0 | Manipulación básica de imágenes | [PyPI - Pillow](https://pypi.org/project/Pillow/) | `pip install Pillow>=9.0.0` |
| **scikit-learn** | 1.0.0 | Algoritmos de aprendizaje automático auxiliares | [PyPI - scikit-learn](https://pypi.org/project/scikit-learn/) | `pip install scikit-learn>=1.0.0` |
| **scikit-image** | 0.19.0 | Algoritmos de procesamiento de imágenes | [PyPI - scikit-image](https://pypi.org/project/scikit-image/) | `pip install scikit-image>=0.19.0` |
| **Matplotlib** | 3.5.0 | Generación de gráficos (uso interno) | [PyPI - matplotlib](https://pypi.org/project/matplotlib/) | `pip install matplotlib>=3.5.0` |
| **Joblib** | 1.1.0 | Serialización eficiente de objetos Python | [PyPI - joblib](https://pypi.org/project/joblib/) | `pip install joblib>=1.1.0` |
| **Protobuf** | (Compatible) | Estructura de datos para TensorFlow | [PyPI - protobuf](https://pypi.org/project/protobuf/) | `pip install protobuf` |

### Notas de Instalación
1. **Entorno Virtual:** Se recomienda encarecidamente usar un entorno virtual (`venv` o `conda`) para evitar conflictos con otras librerías del sistema.
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # En Windows
   source venv/bin/activate  # En Linux/Mac
   ```
2. **Actualización de pip:** Si encuentra errores de instalación, intente actualizar pip primero:
   ```bash
   python -m pip install --upgrade pip
   ```

---

## 📂 Estructura del Proyecto

- `main.py`: Script principal de ejecución.
- `src/`: Código fuente del sistema (núcleo, gui, utilidades).
- `models/`: Archivos de modelos entrenados (`.h5`) y configuraciones.
- `Dataset/`: Imágenes y plantillas para referencia o entrenamiento.
- `output/`: Carpeta donde se guardan los resultados de las ejecuciones.
