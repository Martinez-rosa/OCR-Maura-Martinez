"""
Módulo de segmentación de documentos.

Divide la imagen en regiones (Texto, Tablas, Imágenes) y segmenta líneas y caracteres.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import os

class DocumentSegmenter:
    """
    Segmentador de documentos.
    Responsabilidad:
      - Preprocesar la página (gris, equalización, binarización adaptativa).
      - Separar regiones densas compatibles con imágenes y conservar texto.
      - Extraer líneas de texto mediante clustering DBSCAN (o método legacy).
      - Extraer caracteres por contornos y normalizarlos a 32x32 con padding.
    Atributos:
      debug: activa guardado de caracteres y trazas.
      east_model/east_path: soporte desactivado para detector EAST (opcional).
    Métodos:
      segment_page(img): retorna dict con text_lines e imagen binaria.
      _extract_text_lines(binary_text): clustering por Y.
      _extract_text_lines_legacy(binary_text): método morfológico clásico.
      segment_characters(line_img): contornos ordenados izquierda→derecha.
      _make_square(img): normaliza a cuadrado 32x32 con margen.
    """
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.east_model = None
        self.east_path = os.path.join("models", "frozen_east_text_detection.pb")

    def segment_page(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Segmenta la página completa en regiones.
        Retorna diccionario con recortes de tablas, imágenes y líneas de texto.
        """
        # 1. Preprocesamiento básico para segmentación
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        
        # --- MEJORA: Binarización Adaptativa ---
        # Otsu falla con iluminación desigual. Adaptive Threshold es mejor para documentos escaneados/fotos.
        # Usamos blockSize=25 (tamaño ventana) y C=10 (constante a restar)
        binary = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
        
        # Limpieza de ruido (puntos pequeños)
        # kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)
        
        # 2. Detectar Tablas (DESACTIVADO PARA PRIORIZAR TEXTO)
        # El usuario solicitó explícitamente "solo texto, no tablas".
        # Saltamos la lógica de eliminación de máscaras de tabla para evitar borrar texto accidentalmente.
        
        tables = []
        tables_rects = []
        
        # Usamos la imagen binaria completa como máscara de texto inicial
        text_mask = binary.copy()
        final_text_mask = text_mask.copy()

        # (Código de detección de tablas original comentado o omitido para simplificar)


        cnn_boxes = []
        # Desactivamos EAST para evitar que filtre texto válido que el modelo no detecte.
        # Confiamos en la segmentación morfológica tradicional.
        # final_text_mask = text_mask.copy()
        
        # if os.path.exists(self.east_path):
        #     ... (Código EAST deshabilitado) ...


        kernel_img = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        img_candidates = cv2.dilate(binary, kernel_img, iterations=2)
        mask_non_text = cv2.bitwise_and(img_candidates, cv2.bitwise_not(final_text_mask))
        contours, _ = cv2.findContours(mask_non_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        images = []
        codes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 3000:
                continue
            roi = binary[y:y+h, x:x+w]
            density = cv2.countNonZero(roi) / max(area, 1)
            sub_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_small = sum(1 for c in sub_contours if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] < 500)
            if density > 0.5 and num_small < 5:
                images.append(img[y:y+h, x:x+w])
                cv2.rectangle(final_text_mask, (x, y), (x+w, y+h), 0, -1)
                cv2.rectangle(text_mask, (x, y), (x+w, y+h), 0, -1)
        
        # 4. Extraer Líneas de Texto
        text_lines = self._extract_text_lines(text_mask)
        
        return {
            "tables": tables,
            "tables_rects": tables_rects,
            "images": images,
            "codes": codes,
            "text_lines": text_lines,
            "binary_image": binary # Útil para debug o extracción fina
        }

    def _extract_text_lines(self, binary_text: np.ndarray) -> List[np.ndarray]:
        """
        Extrae líneas de texto usando Clustering (DBSCAN) sobre componentes conectados.
        Esto es más robusto para manuscritos y líneas no perfectamente horizontales.
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            print("[WARN] sklearn no encontrado, usando método tradicional.")
            return self._extract_text_lines_legacy(binary_text)

        # 1. Encontrar todos los componentes (letras/palabras)
        # Dilatar un poco para unir letras en palabras, pero NO tanto como para unir líneas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
        dilated = cv2.dilate(binary_text, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 5 and h > 8: # Filtrar ruido
                # Guardamos centroide Y para clustering
                cy = y + h // 2
                bboxes.append({'bbox': (x, y, w, h), 'cy': cy})
        
        if not bboxes:
            return []

        # 2. Clustering por coordenada Y (Centros)
        # Convertir a array para sklearn
        Y = np.array([b['cy'] for b in bboxes]).reshape(-1, 1)
        
        # DBSCAN: eps es la distancia máxima entre puntos para considerarlos del mismo cluster (misma línea)
        # min_samples=1 para que incluso un punto aislado sea un cluster (línea de una sola palabra)
        # eps debería ser aprox la mitad de la altura de una línea.
        # Ajustado a 35 para manuscrito con espaciado irregular.
        clustering = DBSCAN(eps=35, min_samples=1).fit(Y)
        labels = clustering.labels_
        
        # 3. Agrupar bboxes por cluster (Línea)
        lines_dict = {}
        for i, label in enumerate(labels):
            if label not in lines_dict:
                lines_dict[label] = []
            lines_dict[label].append(bboxes[i]['bbox'])
            
        # 4. Construir imágenes de línea
        lines = []
        # Ordenar clusters por posición Y promedio
        sorted_labels = sorted(lines_dict.keys(), key=lambda l: np.mean([b[1] for b in lines_dict[l]]))
        
        for label in sorted_labels:
            group = lines_dict[label]
            if not group: continue
            
            # Encontrar bounding box total de la línea
            min_x = min(b[0] for b in group)
            min_y = min(b[1] for b in group)
            max_x = max(b[0] + b[2] for b in group)
            max_y = max(b[1] + b[3] for b in group)
            
            # Margen
            y0, y1 = max(0, min_y - 4), min(binary_text.shape[0], max_y + 4)
            x0, x1 = max(0, min_x - 4), min(binary_text.shape[1], max_x + 4)
            
            line_img = binary_text[y0:y1, x0:x1]
            lines.append(line_img)
            
        return lines

    def _extract_text_lines_legacy(self, binary_text: np.ndarray) -> List[np.ndarray]:
        """
        Extrae líneas de texto usando proyección horizontal o contornos dilatados (Método Legacy).
        """
        # Dilatar horizontalmente para conectar letras en palabras y palabras en líneas
        # Aumentamos el kernel para asegurar que palabras separadas se unan en una sola línea
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        dilated = cv2.dilate(binary_text, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        # Filtrar ruido muy pequeño (puntos aislados)
        bounding_boxes = [b for b in bounding_boxes if b[2] > 5 and b[3] > 8]
        
        # Ordenar por coordenada Y (arriba a abajo)
        bounding_boxes.sort(key=lambda b: b[1])
        
        lines = []
        for x, y, w, h in bounding_boxes:
            # Extraer la región del binario original (sin dilatar)
            # Agregamos un pequeño margen
            y0, y1 = max(0, y-2), min(binary_text.shape[0], y+h+2)
            x0, x1 = max(0, x-2), min(binary_text.shape[1], x+w+2)
            line_img = binary_text[y0:y1, x0:x1]
            lines.append(line_img)
            
        return lines


    def segment_characters(self, line_img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segmenta caracteres individuales de una línea de texto.
        """
        # Eliminar dilatación que fusiona caracteres cercanos
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        # dilated = cv2.dilate(line_img, kernel, iterations=1)
        
        # Encontrar contornos directamente sobre la imagen de línea (copia para evitar modificación)
        contours, _ = cv2.findContours(line_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        chars_data = []
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        # Ordenar de izquierda a derecha
        bounding_boxes.sort(key=lambda b: b[0])
        
        debug_dir = "debug_chars"
        
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            # Filtro de ruido: Si es demasiado pequeño, lo ignoramos
            # Ajuste: Bajamos w mínimo a 2 para permitir 'i', 'l', '1' delgados.
            # Ajuste: Subimos h mínimo a 10 para evitar ruido de puntos sueltos.
            if w < 2 or h < 10: 
                continue 
            
            # Recorte del original
            char_img = line_img[y:y+h, x:x+w]
            
            # Padding cuadrado y resize
            char_img_sq = self._make_square(char_img)
            
            if self.debug and os.path.exists(debug_dir):
                import time
                timestamp = int(time.time() * 100000)
                fname = os.path.join(debug_dir, f"char_{timestamp}_{i}.png")
                cv2.imwrite(fname, char_img_sq)

            chars_data.append({
                'img': char_img_sq,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
            
        return chars_data

    def _make_square(self, img: np.ndarray, size: int = 32) -> np.ndarray:
        """
        Redimensiona la imagen a size x size manteniendo aspect ratio (padding).
        Añade un borde extra para evitar que toque los bordes (target 28x28).
        """
        h, w = img.shape
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

    def extract_table_structure(self, table_bin: np.ndarray) -> List[List[np.ndarray]]:
        """
        Intenta segmentar celdas de una tabla binaria.
        Retorna lista de filas, donde cada fila es lista de imágenes de celdas.
        """
        # 1. Detectar filas (proyección horizontal)
        # Usamos un kernel muy ancho para detectar líneas horizontales divisorias
        kernel_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, table_bin.shape[1] // 2), 1))
        lines_hor = cv2.erode(table_bin, kernel_hor, iterations=1)
        lines_hor = cv2.dilate(lines_hor, kernel_hor, iterations=1)
        
        # Encontrar coordenadas Y de las líneas
        contours_hor, _ = cv2.findContours(lines_hor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        y_coords = sorted([cv2.boundingRect(c)[1] for c in contours_hor])
        
        # Si no hay líneas claras, intentamos heurística de espacios vacíos
        if len(y_coords) < 2:
            return [] # No se pudo detectar estructura
            
        rows = []
        for i in range(len(y_coords) - 1):
            y1 = y_coords[i]
            y2 = y_coords[i+1]
            if y2 - y1 < 10: continue # Fila muy fina
            
            row_slice = table_bin[y1:y2, :]
            
            # 2. Detectar columnas en esta fila (proyección vertical)
            # Invertimos lógica: buscamos espacios verticales vacíos o líneas verticales
            # Simplificación: Segmentar por contornos externos en la fila
            # (Asumiendo que las celdas tienen contenido separado)
            
            # Una mejor aproximación para tablas con bordes es usar las líneas verticales detectadas antes
            # Pero aquí haremos segmentación por "bloques de texto" dentro de la fila
            
            cell_contours, _ = cv2.findContours(row_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cell_bboxes = [cv2.boundingRect(c) for c in cell_contours]
            cell_bboxes.sort(key=lambda b: b[0])
            
            row_cells = []
            for cx, cy, cw, ch in cell_bboxes:
                if cw > 5 and ch > 5:
                    row_cells.append(row_slice[cy:cy+ch, cx:cx+cw])
            
            if row_cells:
                rows.append(row_cells)
                
        return rows
