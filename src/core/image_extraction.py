import cv2
import numpy as np
import os
import json
from typing import List, Dict, Any

class ImageExtractor:
    """
    Detecta y extrae figuras/fotos dentro de un documento.
    Enfoque:
      - Gradientes (Sobel) para captar detalle visual.
      - Suavizado + Otsu para binarizar zonas densas.
      - Cierre morfológico para formar bloques compactos.
      - Filtrado por área/ratio y desviación estándar para distinguir
        fondos planos vs. imágenes complejas/diagramas.
    Guarda recortes en output/images y metadatos JSON para trazabilidad.
    """
    def __init__(self, min_area: int = 2000, min_w: int = 30, min_h: int = 30):
        """
        Inicializa el extractor de imágenes.
        :param min_area: Área mínima para considerar una región como imagen.
        :param min_w: Ancho mínimo.
        :param min_h: Alto mínimo.
        """
        self.min_area = min_area
        self.min_w = min_w
        self.min_h = min_h

    def extract(self, image: np.ndarray, output_dir: str, base_filename: str) -> List[Dict[str, Any]]:
        """
        Detecta y extrae imágenes de un documento.
        :param image: Imagen original (numpy array).
        :param output_dir: Directorio donde guardar las imágenes extraídas.
        :param base_filename: Nombre base para los archivos generados.
        :return: Lista de metadatos de las imágenes extraídas.
        """
        # Asegurar directorio de salida
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Preprocesamiento para detección de regiones
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detección de bordes/gradientes para encontrar zonas de alta densidad (imágenes)
        # El texto tiene alta frecuencia, pero las imágenes suelen ser bloques densos.
        # Usamos Sobel para detectar gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Magnitud del gradiente
        gradient = cv2.magnitude(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)
        
        # Difuminar para conectar componentes cercanos y reducir ruido de texto
        blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
        
        # Binarizar: Zonas con mucho detalle (bordes) se vuelven blancas
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas para cerrar huecos y formar bloques rectangulares
        # Usamos un kernel grande rectangular para fusionar el contenido de la imagen
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        extracted_metadata = []
        count = 1
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            # Filtros básicos
            if area > self.min_area and w > self.min_w and h > self.min_h:
                # Chequeo de relación de aspecto (evitar líneas muy finas o columnas de texto muy estrechas)
                aspect_ratio = w / float(h)
                if aspect_ratio > 15 or aspect_ratio < 0.1:
                     continue
                     
                # Extraer ROI (Región de Interés) de la imagen ORIGINAL
                roi = image[y:y+h, x:x+w]
                
                # --- HEURÍSTICA TEXTO vs IMAGEN ---
                # Las fotos suelen tener una varianza de gris más "rica" que el texto (que es bimodal).
                roi_gray = gray[y:y+h, x:x+w]
                mean, std_dev = cv2.meanStdDev(roi_gray)
                
                # Si la desviación estándar es muy baja, es un bloque de color plano (fondo/rectángulo simple)
                if std_dev < 10: 
                    continue
                
                # Clasificación simple (Bonus)
                img_type = "unknown"
                if std_dev > 40:
                    img_type = "photo_complex"
                else:
                    img_type = "diagram_or_simple"

                # Guardar imagen
                filename = f"{base_filename}_img_{count:02d}.png"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, roi)
                
                print(f"[EXTRACT] Imagen detectada: {filename} (Area: {area}, Tipo: {img_type})")
                
                extracted_metadata.append({
                    "id": count,
                    "filename": filename,
                    "path": filepath,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "type": img_type
                })
                count += 1
                
        # Guardar metadatos
        if extracted_metadata:
            meta_path = os.path.join(output_dir, f"{base_filename}_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(extracted_metadata, f, indent=4)
                
        return extracted_metadata
