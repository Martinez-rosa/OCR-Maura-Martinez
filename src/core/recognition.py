"""
Módulo de reconocimiento de caracteres mediante Template Matching (Correlación).

Compara cada carácter segmentado contra una base de datos de plantillas (prototipos)
y devuelve el carácter con mayor similitud. No utiliza Machine Learning ni entrenamiento.
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models

class TemplateMatcher:
    """
    Reconocedor por plantillas (sin ML).
    - Carga imágenes prototipo desde Dataset (soporta estructura may/min).
    - Normaliza polaridad, recorta y ajusta a 32x32 con padding.
    - Compara caracteres de entrada con correlación normalizada y
      penalización suave por aspect ratio para manuscrito.
    Atributos:
      templates_dir: raíz del dataset de referencia.
      templates: dict label -> lista de plantillas normalizadas.
      template_ratios: dict label -> ratios para penalización adaptativa.
    """
    def __init__(self, templates_dir: str):
        self.templates_dir = templates_dir
        self.templates: Dict[str, List[np.ndarray]] = {}
        self.template_ratios: Dict[str, List[float]] = {}
        self.load_templates()

    def load_templates(self):
        """
        Carga las plantillas recursivamente.
        Usa el nombre de la carpeta contenedora inmediata como la etiqueta del carácter.
        """
        if not os.path.exists(self.templates_dir):
            print(f"[WARN] Directorio de plantillas no encontrado: {self.templates_dir}")
            return

        print("[INFO] Cargando plantillas...", end=" ")
        count = 0
        # Recorrer recursivamente para soportar estructuras anidadas (ej. templates/mayusculas/A)
        for root, dirs, files in os.walk(self.templates_dir):
            label_name = os.path.basename(root)
            
            # --- NUEVA LÓGICA PARA DATASET (may/min) ---
            parent_name = os.path.basename(os.path.dirname(root))
            if label_name == "may":
                label = parent_name.upper()
                is_valid = True
            elif label_name == "min":
                label = parent_name.lower()
                is_valid = True
            else:
                # --- LÓGICA LEGACY ---
                # Evitar usar el directorio raíz de templates como etiqueta
                if os.path.abspath(root) == os.path.abspath(self.templates_dir):
                    continue
                
                # === FILTRO DE SÍMBOLOS ===
                allowed_specials = ["at", "underscore", "hyphen"] 
                ignored_labels = ["dot", "colon", "semicolon", "comma", "slash", "backslash", "quote", "doublequote"]

                is_valid = False
                if label_name in ignored_labels:
                    is_valid = False
                elif label_name.isalnum():
                    is_valid = True
                elif label_name in allowed_specials:
                    is_valid = True
                
                # Excepciones: Si es una carpeta contenedora como "mayusculas", la ignoramos
                if label_name in ["mayusculas", "minusculas", "numeros", "simbolos", "digital", "manuscrito", "Dataset"]:
                    is_valid = False

                if is_valid:
                    # Mapeo de nombres de carpeta a caracteres reales
                    if label_name == "at": label = "@"
                    elif label_name == "dot": label = "."
                    elif label_name == "underscore": label = "_"
                    elif label_name == "hyphen": label = "-"
                    else: label = label_name

            if not is_valid:
                continue
            
            if label not in self.templates:
                self.templates[label] = []

            # Límite de plantillas por clase para optimizar rendimiento
            # Si hay demasiadas (ej. 100 por letra), el sistema tardará minutos en arrancar
            # y segundos en reconocer CADA letra.
            MAX_TEMPLATES_PER_CLASS = 15
            current_count = len(self.templates[label])
            if current_count >= MAX_TEMPLATES_PER_CLASS:
                continue

            for filename in files:
                if current_count >= MAX_TEMPLATES_PER_CLASS:
                    break

                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(root, filename)
                    
                    try:
                        # Leer imagen
                        stream = open(img_path, "rb")
                        bytes = bytearray(stream.read())
                        numpyarray = np.asarray(bytes, dtype=np.uint8)
                        img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
                        stream.close()
                    except Exception:
                        continue
                    
                    if img is None: continue
                    
                    # Procesamiento idéntico al input: Binarizar Otsu + Invertir si es necesario
                    # Las plantillas generadas son letras negras sobre blanco (normalmente)
                    # o blancas sobre negro. Debemos estandarizar a blanco sobre negro.
                    
                    # Binarizar
                    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Chequeo de polaridad: Queremos Texto=255 (Blanco), Fondo=0 (Negro)
                    # Contamos píxeles blancos
                    n_white = cv2.countNonZero(bin_img)
                    n_pixels = bin_img.shape[0] * bin_img.shape[1]
                    
                    # Si hay más blanco que negro, es fondo blanco -> Invertir
                    if n_white > n_pixels // 2:
                        bin_img = cv2.bitwise_not(bin_img)
                    
                    # Recortar al contenido (ROI)
                    coords = cv2.findNonZero(bin_img)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        bin_img = bin_img[y:y+h, x:x+w]
                    else:
                        # Si no hay contenido (todo negro), saltar
                        continue
                    
                    # Calcular aspect ratio original
                    ratio = w / h if h > 0 else 1.0

                    # Redimensionar (Forzando ajuste)
                    resized = self._resize_template(bin_img)
                    
                    if label not in self.templates:
                        self.templates[label] = []
                    self.templates[label].append(resized)
                    
                    if label not in self.template_ratios:
                        self.template_ratios[label] = []
                    self.template_ratios[label].append(ratio)

                    count += 1
                    current_count += 1
        
        print(f"OK. {count} plantillas cargadas para {len(self.templates)} clases.")

    def _resize_template(self, img: np.ndarray, size: int = 32) -> np.ndarray:
        """Redimensiona manteniendo aspect ratio y rellenando con negro (padding).
        Añade un borde extra para evitar plantillas sólidas."""
        h, w = img.shape[:2]
        if h == 0 or w == 0: return np.zeros((size, size), dtype=np.uint8)
        
        # Target inner size (margen de 2px)
        target_size = size - 4
        
        # Calcular escala
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Asegurar dimensiones mínimas de 1px
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        
        # Redimensionar contenido
        # Si escalamos hacia arriba (scale > 1), usar CUBIC. Si es hacia abajo, AREA.
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
        # Crear imagen negra cuadrada
        square = np.zeros((size, size), dtype=np.uint8)
        
        # Centrar
        y_offset = (size - new_h) // 2
        x_offset = (size - new_w) // 2
        
        square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return square

    def recognize_char(self, char_img: np.ndarray, original_wh: Tuple[int, int] = None) -> Tuple[str, float]:
        """
        Compara la imagen del caracter con todas las plantillas.
        Usa variantes morfológicas y suavizado para ser más robusto.
        """
        best_score = -1.0
        best_label = "?"
        
        # Optimización: Si la imagen está vacía
        if cv2.countNonZero(char_img) == 0:
            return "", 0.0

        # Preprocesar input: Crear variantes para manejar diferencias de grosor y ruido
        # 1. Normal
        img_normal = char_img
        # 2. Suavizada (Blur) - ayuda con pequeñas desalineaciones
        img_blur = cv2.GaussianBlur(char_img, (3, 3), 0)
        # 3. Dilatada (más gruesa) - si el input es muy fino
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_dilated = cv2.dilate(char_img, kernel, iterations=1)
        # 4. Erosionada (más fina) - si el input es muy grueso
        img_eroded = cv2.erode(char_img, kernel, iterations=1)
        
        variants = [img_normal, img_blur, img_dilated, img_eroded]

        # Iterar sobre todas las clases
        for label, templates_list in self.templates.items():
            for idx, templ in enumerate(templates_list):
                
                # Suavizar también el template ligeramente
                templ_blur = cv2.GaussianBlur(templ, (3, 3), 0)
                
                # Probar contra todas las variantes del input y quedarse con la mejor
                local_best_score = -1.0
                
                for variant in variants:
                    # Usamos correlación normalizada
                    res = cv2.matchTemplate(variant, templ_blur, cv2.TM_CCOEFF_NORMED)
                    score = res[0][0]
                    if score > local_best_score:
                        local_best_score = score
                
                score = local_best_score
                
                # Penalización por aspect ratio (suavizada para manuscrito)
                if original_wh and original_wh[1] > 0:
                    input_ratio = original_wh[0] / original_wh[1]
                    
                    if label in self.template_ratios and idx < len(self.template_ratios[label]):
                        tmpl_ratio = self.template_ratios[label][idx]
                        diff = abs(input_ratio - tmpl_ratio)
                        
                        # Penalización adaptativa (menos severa para manuscrito)
                        if diff > 0.6: # Antes 0.4
                            score -= diff * 0.4
                        elif diff > 0.3: # Antes 0.15
                            score -= diff * 0.15
                
                if score > best_score:
                    best_score = score
                    best_label = label
        
        # === LÓGICA DE DECISIÓN AVANZADA ===
        
        # 1. Penalización para símbolos
        if best_label in [".", "-", "_", "@", ",", ":", ";"]:
            if best_score < 0.6: # Antes 0.75
                pass 
        
        # 2. Umbral mínimo global (reducido para manuscrito)
        if best_score < 0.35: # Antes 0.45
            return "?", best_score
            
        return best_label, best_score

    def recognize_line(self, chars_data: List[Dict]) -> str:
        """
        Convierte una lista de datos de caracteres en un string.
        Inserta espacios basándose en la distancia horizontal.
        """
        if not chars_data:
            return ""
            
        text = ""
        # Calcular ancho promedio de caracteres para estimar el espacio
        widths = [c['w'] for c in chars_data]
        avg_width = np.mean(widths) if widths else 10
        
        # Umbral para espacio: 40% del ancho promedio
        space_threshold = avg_width * 0.4
        
        for i, char_data in enumerate(chars_data):
            char_img = char_data['img']
            
            # Espacios
            if i > 0:
                prev_char = chars_data[i-1]
                gap = char_data['x'] - (prev_char['x'] + prev_char['w'])
                if gap > space_threshold:
                    text += " "
            
            orig_w = char_data.get('w', 0)
            orig_h = char_data.get('h', 0)
            label, score = self.recognize_char(char_img, original_wh=(orig_w, orig_h))
            
            # Solo añadir si no es basura
            if label != "?":
                text += label
            
        return self._refine_text_context(text)

    def _refine_text_context(self, text: str) -> str:
        """
        Aplica heurísticas simples para corregir confusiones comunes (0 vs O, 1 vs I, etc.)
        basándose en el contexto de la palabra (si es mayormente letras o números).
        """
        words = text.split(" ")
        refined_words = []
        
        for word in words:
            # Contar dígitos y letras
            n_digits = sum(c.isdigit() for c in word)
            n_alpha = sum(c.isalpha() for c in word)
            length = len(word)
            
            if length == 0:
                refined_words.append("")
                continue
            
            new_word = list(word)
            
            # Caso 1: Palabra mayormente alfabética (ej. "HOLA") -> corregir números que parecen letras
            # Umbral: Más del 50% son letras y tiene algún dígito
            if n_alpha > n_digits:
                for i, char in enumerate(new_word):
                    if char == '0': new_word[i] = 'o'
                    elif char == '6': new_word[i] = 'e' # e vs 6 es común en este dataset
                    elif char == '1': new_word[i] = 'l' if i > 0 else 'I' # l intermedia, I inicial
                    elif char == '5': new_word[i] = 's'
                    elif char == '8': new_word[i] = 'B'
                    elif char == '2': new_word[i] = 'z'
            
            # Caso 2: Palabra mayormente numérica (ej. "2023") -> corregir letras que parecen números
            elif n_digits > n_alpha:
                for i, char in enumerate(new_word):
                    if char.lower() == 'o': new_word[i] = '0'
                    elif char == 'l' or char == 'I': new_word[i] = '1'
                    elif char.lower() == 's': new_word[i] = '5'
                    elif char == 'B': new_word[i] = '8'
                    elif char.lower() == 'z': new_word[i] = '2'
            
            refined_words.append("".join(new_word))
            
        return " ".join(refined_words)

import pickle

class CNNRecognizer:
    """
    Reconocedor basado en modelos Keras (.h5).
    - Carga el modelo indicado por `model_path` y su mapeo de clases
      (char_mapping_digital.pkl / char_mapping_manuscrito.pkl).
    - Convierte secuencias de caracteres segmentados en texto usando
      el modelo y los mapeos, ofreciendo mayor robustez que plantillas.
    Atributos:
      templates_dir: ruta del dataset (para utilidades).
      model_path: ruta del modelo Keras a utilizar.
      model: instancia cargada de tf.keras si existe.
      classes: lista/mapeo de clases para inferencia.
    """
    def __init__(self, templates_dir: str, model_path: str = "models/cnn_ocr.h5"):
        self.templates_dir = templates_dir
        self.model_path = model_path
        self.model = None
        self.classes: List[str] = []
        
        # Intentar cargar modelo existente
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                
                # Intentar cargar mapeo .pkl (prioridad)
                mapping_path = self.model_path.replace(".h5", "") + "_mapping.pkl"
                # Compatibilidad con nombre solicitado por usuario: char_mapping_digital.pkl si el modelo es model_digital.h5
                # Vamos a inferir el nombre del mapping basado en el nombre del modelo
                # Si model_path es "models/model_digital.h5", mapping debería ser "models/char_mapping_digital.pkl"
                
                base_name = os.path.basename(self.model_path)
                if "digital" in base_name:
                    mapping_name = "char_mapping_digital.pkl"
                elif "manuscrito" in base_name:
                    mapping_name = "char_mapping_manuscrito.pkl"
                else:
                    mapping_name = base_name.replace(".h5", "_mapping.pkl")
                
                mapping_path = os.path.join(os.path.dirname(self.model_path), mapping_name)

                if os.path.exists(mapping_path):
                    with open(mapping_path, "rb") as f:
                        self.classes = pickle.load(f)
                else:
                    # Fallback a .labels.txt
                    meta_path = self.model_path + ".labels.txt"
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            self.classes = [line.strip() for line in f if line.strip()]
            except Exception:
                self.model = None

        # Si no hay modelo, intentar entrenar (pero solo si se llama explícitamente o es el default)
        # Para evitar reentrenamientos accidentales en producción, mejor dejar que el script de entrenamiento lo haga.
        # Pero mantendremos la lógica original de intentar entrenar si no existe, por robustez.
        if self.model is None and os.path.exists(self.templates_dir):
            pass

    def _enumerate_templates(self) -> Tuple[List[np.ndarray], List[int], List[str], Dict[str, int]]:
        X: List[np.ndarray] = []
        y: List[int] = []
        labels: List[str] = []
        label_to_idx: Dict[str, int] = {}
        
        if not os.path.exists(self.templates_dir):
            return [], [], [], {}
            
        ignored_containers = ["mayusculas", "minusculas", "numeros", "simbolos", "digital", "manuscrito", "templates"]
        ignored_labels = ["dot", "colon", "semicolon", "comma", "slash", "backslash", "quote", "doublequote"]
        
        print(f"Buscando templates en: {os.path.abspath(self.templates_dir)}")
        for root, dirs, files in os.walk(self.templates_dir):
            # Normalizar separadores
            root_norm = root.replace("\\", "/")
            label_name = os.path.basename(root)
            
            # --- NUEVA LÓGICA PARA DATASET (may/min) ---
            # Si estamos en una carpeta "may" o "min", el label es el padre
            parent_name = os.path.basename(os.path.dirname(root))
            
            if label_name == "may":
                label = parent_name.upper()
            elif label_name == "min":
                label = parent_name.lower()
            else:
                # --- LÓGICA LEGACY (Manuscrito / Estructura antigua) ---
                label = label_name
                
                # Ignorar raíz
                if os.path.abspath(root) == os.path.abspath(self.templates_dir):
                    continue
                    
                # Ignorar contenedores si no son labels válidos
                if label_name in ignored_containers:
                    continue
            
            # Ignorar etiquetas no deseadas (slash, etc.) si estamos en modo legacy
            # En modo Dataset, "may" y "min" ya han sido procesados, pero "label" ahora es "A" o "a"
            if label in ignored_labels:
                continue

            # Mapeos especiales (solo aplica si label es el nombre de la carpeta, ej. "at")
            if label == "at": label = "@"
            elif label == "underscore": label = "_"
            elif label == "hyphen": label = "-"
            elif label == "lparen": label = "("
            elif label == "rparen": label = ")"
            elif label == "plus": label = "+"
            
            # Verificar si tiene imágenes
            has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for f in files)
            if not has_images:
                continue
            
            # Permitir que el nombre de la carpeta sea el label directo (ej. "(", "+") si el sistema de archivos lo permite
            
            if label not in label_to_idx:
                label_to_idx[label] = len(labels)
                labels.append(label)
                
            for filename in files:
                if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    continue
                img_path = os.path.join(root, filename)
                try:
                    stream = open(img_path, "rb")
                    bytes_data = bytearray(stream.read())
                    numpyarray = np.asarray(bytes_data, dtype=np.uint8)
                    img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
                    stream.close()
                except Exception:
                    continue
                if img is None:
                    continue
                    
                # Preprocesamiento
                _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                n_white = cv2.countNonZero(bin_img)
                n_pixels = bin_img.shape[0] * bin_img.shape[1]
                if n_white > n_pixels // 2:
                    bin_img = cv2.bitwise_not(bin_img)
                    
                coords = cv2.findNonZero(bin_img)
                if coords is None:
                    continue
                    
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(coords)
                roi = bin_img[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect]
                roi = self._make_square(roi, 32)
                
                X.append(roi.astype(np.float32) / 255.0)
                y.append(label_to_idx[label])
                
        return X, y, labels, label_to_idx

    def _enumerate_dataset(self, dataset_dir: str, X: List, y: List, labels: List, label_to_idx: Dict):
        """Appends data from the new dataset structure into the provided lists and maps."""
        if not os.path.exists(dataset_dir):
            return

        print(f"Buscando en dataset extra: {os.path.abspath(dataset_dir)}")
        # Structure: dataset_dir/{char}/{case}/image.png
        for char_dir in os.scandir(dataset_dir):
            if not char_dir.is_dir():
                continue
            
            label_char = char_dir.name
            if label_char == "n_": label_char = "ñ"
            
            for case_dir in os.scandir(char_dir.path):
                if not case_dir.is_dir() or case_dir.name not in ["may", "min"]:
                    continue

                current_label = label_char.upper() if case_dir.name == "may" else label_char.lower()

                if current_label not in label_to_idx:
                    label_to_idx[current_label] = len(labels)
                    labels.append(current_label)
                
                label_idx = label_to_idx[current_label]

                for file_entry in os.scandir(case_dir.path):
                    if not file_entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue
                    
                    img_path = file_entry.path
                    try:
                        stream = open(img_path, "rb")
                        bytes_data = bytearray(stream.read())
                        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
                        img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
                        stream.close()
                    except Exception:
                        continue
                    if img is None:
                        continue

                    # Preprocesamiento
                    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    n_white = cv2.countNonZero(bin_img)
                    n_pixels = bin_img.shape[0] * bin_img.shape[1]
                    if n_white > n_pixels // 2:
                        bin_img = cv2.bitwise_not(bin_img)

                    coords = cv2.findNonZero(bin_img)
                    if coords is None:
                        continue
                    
                    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(coords)
                    roi = bin_img[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect]
                    roi = self._make_square(roi, 32)
                    
                    X.append(roi.astype(np.float32) / 255.0)
                    y.append(label_idx)

    def _make_square(self, img: np.ndarray, size: int = 32) -> np.ndarray:
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((size, size), dtype=np.uint8)
        target = size - 4
        scale = min(target / h, target / w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(img, (nw, nh), interpolation=interp)
        sq = np.zeros((size, size), dtype=np.uint8)
        y0 = (size - nh) // 2
        x0 = (size - nw) // 2
        sq[y0:y0+nh, x0:x0+nw] = resized
        return sq

    def _build_model(self, num_classes: int) -> tf.keras.Model:
        inputs = layers.Input(shape=(32, 32, 1))
        x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def train_from_templates(self, epochs: int = 5, extra_dataset_dir: str = None):
        X_list, y_list, labels, label_to_idx = self._enumerate_templates()

        if extra_dataset_dir:
            self._enumerate_dataset(extra_dataset_dir, X_list, y_list, labels, label_to_idx)

        if not X_list:
            print("No se encontraron muestras para entrenar.")
            return

        X = np.expand_dims(np.stack(X_list, axis=0), axis=-1)
        y = np.asarray(y_list, dtype=np.int64)

        if X.shape[0] == 0:
            return
        print(f"Entrenando con {X.shape[0]} muestras y {len(labels)} clases.")
        self.classes = labels
        
        # Mezclar datos (Shuffle) antes de entrenar para que la validación sea representativa
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        self.model = self._build_model(num_classes=len(labels))
        
        # --- DATA AUGMENTATION ---
        # Crucial para manuscrito donde la orientación, zoom y forma varían mucho
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            datagen = ImageDataGenerator(
                rotation_range=20,      # Rotación +/- 20 grados
                width_shift_range=0.2, # Desplazamiento horizontal
                height_shift_range=0.2,# Desplazamiento vertical
                shear_range=0.2,       # Inclinación (cursiva)
                zoom_range=0.2,        # Zoom
                fill_mode='constant',
                cval=0                  # Rellenar con negro
            )
            # Entrenar con generador
            # Dividimos X, y en train/val manualmente para usar fit con generator
            split_idx = int(len(X) * 0.9)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Ajustamos batch_size y steps_per_epoch
            batch_size = 32
            
            print("Iniciando entrenamiento con Data Augmentation...")
            self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                validation_data=(X_val, y_val),
                epochs=epochs,
                steps_per_epoch=len(X_train) // batch_size if len(X_train) >= batch_size else 1,
                verbose=1
            )
        except Exception as e:
            print(f"Advertencia: No se pudo usar Data Augmentation ({e}). Usando fit normal.")
            self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=1, validation_split=0.1)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        
        # Guardar mapping en pickle
        base_name = os.path.basename(self.model_path)
        if "digital" in base_name:
            mapping_name = "char_mapping_digital.pkl"
        elif "manuscrito" in base_name:
            mapping_name = "char_mapping_manuscrito.pkl"
        else:
            mapping_name = base_name.replace(".h5", "_mapping.pkl")
        
        mapping_path = os.path.join(os.path.dirname(self.model_path), mapping_name)
        
        with open(mapping_path, "wb") as f:
            pickle.dump(self.classes, f)
        print(f"Modelo guardado en {self.model_path}")
        print(f"Mapping guardado en {mapping_path}")

    def predict_char(self, char_img: np.ndarray) -> Tuple[str, float]:
        if self.model is None:
            return "?", 0.0
        x = char_img.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=(0, -1))
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])

    def recognize_line(self, chars_data: List[Dict]) -> str:
        if not chars_data:
            return ""
        widths = [c['w'] for c in chars_data]
        avg_width = np.mean(widths) if widths else 10
        space_threshold = avg_width * 0.4
        text = ""
        
        # Filtro estricto de símbolos
        # Permitimos solo alfanuméricos y espacios, y quizas puntos/comas básicos si tienen alta confianza
        ignored_symbols = ["@", "_", "-", "+", "(", ")", "[", "]", "{", "}", "<", ">", "|", "\\", "/", "*", "=", "#", "$", "%", "^", "&"]

        for i, c in enumerate(chars_data):
            if i > 0:
                prev = chars_data[i-1]
                gap = c['x'] - (prev['x'] + prev['w'])
                if gap > space_threshold:
                    text += " "
            label, score = self.predict_char(c['img'])
            
            # FILTRADO DE SIMBOLOS
            if label in ignored_symbols:
                continue # Ignorar símbolo
            
            # Si no es alfanumérico (ej. es un punto, coma, etc.) requerir confianza muy alta
            if not label.isalnum():
                 if score < 0.95: # Muy estricto para símbolos
                     continue

            if label != "?":
                text += label
        return text
