"""
Script de debug para visualizar la segmentación con la arquitectura actual
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from src.core.segmentation import DocumentSegmenter
from src.core.recognition import TemplateMatcher

def debug_segmentation(image_path):
    """Visualiza paso a paso la segmentación y reconocimiento"""
    
    print(f"Procesando: {image_path}")
    
    # Cargar imagen
    if not os.path.exists(image_path):
        print(f"Error: Archivo no encontrado: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar {image_path}")
        return
    
    # Instanciar segmentador y reconocedor
    segmenter = DocumentSegmenter(debug=True)
    matcher = TemplateMatcher(templates_dir="templates")
    
    # 1. SEGMENTACIÓN DE PÁGINA
    print("Ejecutando segmentación de página...")
    regions = segmenter.segment_page(img)
    
    # Recuperar resultados
    tables = regions.get("tables", [])
    images = regions.get("images", [])
    text_lines = regions.get("text_lines", [])
    binary_img = regions.get("binary_image", None)
    
    print(f"✓ Tablas detectadas: {len(tables)}")
    print(f"✓ Imágenes detectadas: {len(images)}")
    print(f"✓ Líneas de texto detectadas: {len(text_lines)}")
    
    # Visualizar detección de regiones sobre la imagen original
    # (Nota: segment_page devuelve los recortes, para visualizar los recuadros
    #  necesitaríamos las coordenadas, que segment_page devuelve parcialmente en tables_rects.
    #  Para debug completo, idealmente segment_page debería devolver rects de todo, 
    #  pero aquí visualizaremos los recortes extraídos).
    
    # Visualizar binaria
    if binary_img is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(binary_img, cmap='gray')
        plt.title("Imagen Binarizada")
        plt.axis('off')
        plt.savefig("debug_binary.png")
        print("✓ Guardado: debug_binary.png")
        plt.close()

    # Visualizar Tablas (si hay)
    if tables:
        fig, axes = plt.subplots(1, len(tables), figsize=(15, 5))
        if len(tables) == 1: axes = [axes]
        for i, tbl in enumerate(tables):
            axes[i].imshow(cv2.cvtColor(tbl, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"Tabla {i+1}")
            axes[i].axis('off')
        plt.savefig("debug_tables.png")
        print("✓ Guardado: debug_tables.png")
        plt.close()

    # Visualizar Imágenes (si hay)
    if images:
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        if len(images) == 1: axes = [axes]
        for i, im in enumerate(images):
            axes[i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"Imagen {i+1}")
            axes[i].axis('off')
        plt.savefig("debug_images.png")
        print("✓ Guardado: debug_images.png")
        plt.close()

    # 2. VISUALIZAR LÍNEAS DE TEXTO
    if text_lines:
        # Mostrar primeras 5 líneas
        n_lines = min(len(text_lines), 10)
        fig, axes = plt.subplots(n_lines, 1, figsize=(10, 2 * n_lines))
        if n_lines == 1: axes = [axes]
        
        for i in range(n_lines):
            line = text_lines[i]
            axes[i].imshow(line, cmap='gray')
            axes[i].set_title(f"Línea {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("debug_lines.png")
        print("✓ Guardado: debug_lines.png")
        plt.close()

    # 3. SEGMENTACIÓN DE CARACTERES (usando todas las líneas para debug)
    print("\nReconociendo texto de las líneas detectadas...")
    
    with open("debug_recognition.txt", "w", encoding="utf-8") as f:
        for i, line_img in enumerate(text_lines):
            # Segmentar caracteres
            chars_data = segmenter.segment_characters(line_img)
            
            # Reconocer
            line_text = matcher.recognize_line(chars_data)
            
            print(f"Línea {i+1}: '{line_text}'")
            f.write(f"Línea {i+1}: {line_text}\n")
            
            # DEBUG ADICIONAL: Analizar 'o' en "Modulo" (Línea 2, segundo caracter)
            if i == 1 and len(chars_data) > 1:
                # El caracter 'o' debería ser el índice 1 (M, o, d, u, l, o)
                # Verificamos posición visual
                char_idx = 1
                char_img = chars_data[char_idx]['img']
                print(f"\n--- DEBUG DETALLADO CARACTER {char_idx} LINEA 2 (Debería ser 'o') ---")
                
                scores = []
                for label, templates_list in matcher.templates.items():
                    best_score_for_label = -1
                    best_templ_idx = -1
                    for idx, templ in enumerate(templates_list):
                        res = cv2.matchTemplate(char_img, templ, cv2.TM_CCOEFF_NORMED)
                        score = res[0][0]
                        
                        # Simular penalización por Aspect Ratio
                        if label in matcher.template_ratios and idx < len(matcher.template_ratios[label]):
                            tmpl_ratio = matcher.template_ratios[label][idx]
                            input_w = chars_data[char_idx]['w']
                            input_h = chars_data[char_idx]['h']
                            input_ratio = input_w / input_h if input_h > 0 else 1.0
                            
                            diff = abs(input_ratio - tmpl_ratio)
                            if diff > 0.4:
                                score -= diff * 0.5
                            elif diff > 0.15:
                                score -= diff * 0.25
                                
                        if score > best_score_for_label:
                            best_score_for_label = score
                            best_templ_idx = idx
                    scores.append((label, best_score_for_label, best_templ_idx))
                
                scores.sort(key=lambda x: x[1], reverse=True)
                print("Top 10 candidatos:")
                for lbl, scr, t_idx in scores[:10]:
                    print(f"  '{lbl}' (tpl {t_idx}): {scr:.4f}")
                print("---------------------------------------")

            # Visualizar primera línea solamente
            if i == 0 and chars_data:
                # Dibujar recuadros sobre la línea
                line_bgr = cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR)
                for char in chars_data:
                    x, y, w, h = char['x'], char['y'], char['w'], char['h']
                    cv2.rectangle(line_bgr, (x, y), (x+w, y+h), (0, 0, 255), 1)
                
                plt.figure(figsize=(15, 3))
                plt.imshow(cv2.cvtColor(line_bgr, cv2.COLOR_BGR2RGB))
                plt.title(f"Segmentación Línea 1 - Texto: {line_text}")
                plt.axis('off')
                plt.savefig("debug_chars_segmentation.png")
                plt.close()
                
                # Guardar caracteres individuales normalizados de la primera línea para inspección visual
                if len(chars_data) > 0:
                     # Crear un grid de imágenes
                     n_chars = len(chars_data)
                     cols = 10
                     rows = (n_chars // cols) + 1
                     fig, axes = plt.subplots(rows, cols, figsize=(15, 2*rows))
                     axes = axes.flatten()
                     for idx, char_data in enumerate(chars_data):
                         char_norm = char_data['img']
                         label, score = matcher.recognize_char(char_norm)
                         axes[idx].imshow(char_norm, cmap='gray')
                         axes[idx].set_title(f"'{label}' ({score:.2f})")
                         axes[idx].axis('off')
                     # Apagar ejes sobrantes
                     for idx in range(n_chars, len(axes)):
                         axes[idx].axis('off')
                     plt.tight_layout()
                     plt.savefig("debug_chars_recognition.png")
                     plt.close()

    print("✓ Guardado: debug_recognition.txt")
    print("✓ Guardado: debug_chars_recognition.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Busca una imagen por defecto si no se pasa argumento
        files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            img_path = files[0]
        else:
            print("Uso: python debug.py <ruta_imagen>")
            sys.exit(1)
            
    debug_segmentation(img_path)

