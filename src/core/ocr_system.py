"""
Sistema OCR Unificado.
Clase de alto nivel que integra segmentación y reconocimiento.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from .segmentation import DocumentSegmenter
from .recognition import TemplateMatcher, CNNRecognizer
from .image_extraction import ImageExtractor

class OCRSystem:
    """
    Orquestador del pipeline OCR.
    - Gestiona segmentación de líneas y caracteres (DocumentSegmenter).
    - Selecciona y utiliza el reconocedor adecuado (CNNRecognizer digital/manuscrito).
    - Opcionalmente extrae figuras/fotos del documento (ImageExtractor).
    - Estructura y guarda resultados en output/run_YYYYMMDD_HHMMSS/.
    Atributos:
      debug: habilita guardado de caracteres y trazas.
      templates_dir: ruta al dataset de referencia (Dataset).
      segmenter: instancia de segmentación de documento.
      image_extractor: instancia para extraer imágenes/diagramas.
      matcher, matcher_manuscrito: respaldo por plantillas (validación).
      cnn_digital, cnn_manuscrito: modelos Keras para reconocimiento.
    """
    def __init__(self, templates_dir: str = "Dataset", debug: bool = False):
        self.debug = debug
        self.templates_dir = templates_dir
        
        # Inicializar componentes
        # Aseguramos que existan las plantillas
        if not os.path.exists(templates_dir):
            print(f"[WARN] Directorio de plantillas '{templates_dir}' no encontrado.")
            
        self.segmenter = DocumentSegmenter(debug=debug)
        self.image_extractor = ImageExtractor() # Nuevo componente
        
        # Matcher para texto DIGITAL (usa raíz de Dataset)
        self.matcher = TemplateMatcher(templates_dir)
        
        # Matcher específico para MANUSCRITO (usa Dataset/manuscrito)
        self.matcher_manuscrito = TemplateMatcher(os.path.join(templates_dir, "manuscrito"))
        
        # Inicializar modelos especializados (Paths actualizados)
        # Nota: CNNRecognizer también necesitará ajustes si usa la estructura de carpetas
        self.cnn_digital = CNNRecognizer(templates_dir=templates_dir, model_path="models/model_digital.h5")
        self.cnn_manuscrito = CNNRecognizer(templates_dir=os.path.join(templates_dir, "manuscrito"), model_path="models/model_manuscrito.h5")
        
    def process_image(self, image_input: Any, text_type: str = "digital", extract_images: bool = False) -> Dict[str, Any]:
        import time
        import datetime
        
        img = None
        if isinstance(image_input, str):
            img_path_str = image_input
            img = self._load_image(image_input)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input
            img_path_str = "uploaded_image"
        else:
            raise ValueError("Input debe ser ruta (str) o imagen (numpy array)")
            
        # Seleccionar modelo
        active_recognizer = None
        
        if text_type == "manuscrito":
            # Usar el reconocedor CNN para manuscrito, que es más potente
            print("[INFO] Usando Reconocimiento con CNN para manuscrito.")
            active_recognizer = self.cnn_manuscrito
        else:
            # Para digital preferimos CNN
            print("[INFO] Usando Reconocimiento con CNN para digital.")
            active_recognizer = self.cnn_digital

        regions = self.segmenter.segment_page(img)

        full_text = ""
        text_lines_content: List[str] = []
        for line_img in regions['text_lines']:
            chars_data = self.segmenter.segment_characters(line_img)
            
            line_text = active_recognizer.recognize_line(chars_data)
            
            if line_text.strip(): # Solo añadir si no es vacío
                full_text += line_text + "\n"
                text_lines_content.append(line_text)

        processed_tables: List[List[List[str]]] = []
        # TABLAS DESACTIVADAS POR SOLICITUD DEL USUARIO
        
        # === GUARDAR RESULTADOS EN DISCO ===
        # Crear carpeta outputs si no existe
        output_base = "output"
        if not os.path.exists(output_base):
            os.makedirs(output_base)
            
        # Carpeta timestamp para esta ejecución
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_base, f"run_{timestamp}")
        os.makedirs(run_dir)
        
        # === NUEVA FUNCIONALIDAD: EXTRACCIÓN DE IMÁGENES ===
        extracted_metadata = []
        if extract_images:
            print("[INFO] Extrayendo imágenes del documento...")
            images_dir = os.path.join(output_base, "images")
            base_name = os.path.splitext(os.path.basename(img_path_str))[0]
            # Usamos un nombre único para evitar colisiones
            unique_base_name = f"{base_name}_{timestamp}"
            
            extracted_metadata = self.image_extractor.extract(img, images_dir, unique_base_name)
            print(f"[INFO] Se extrajeron {len(extracted_metadata)} imágenes en {images_dir}")

        # 1. Guardar Texto
        txt_path = os.path.join(run_dir, "result.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        # 2. Guardar Imágenes extraídas (si las hay) - LEGACY SEGMENTATION IMAGES
        # Mantengo esto por si el segmentador antiguo detectaba algo, pero la nueva funcionalidad es superior.
        extracted_images_paths = []
        for i, img_roi in enumerate(regions.get("images", [])):
            if img_roi is not None and img_roi.size > 0:
                img_name = f"legacy_segment_image_{i}.png"
                img_path = os.path.join(run_dir, img_name)
                cv2.imwrite(img_path, img_roi)

        return {
            "full_text": full_text,
            "text_lines": text_lines_content,
            "extracted_images_count": len(extracted_metadata),
            "extracted_images_metadata": extracted_metadata,
            "output_dir": run_dir
        }

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        try:
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            return cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        except Exception:
            return None
