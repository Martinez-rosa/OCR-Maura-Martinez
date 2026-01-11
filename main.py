"""
Script Principal OCR sin ML.

Orquesta el proceso de OCR utilizando la clase unificada OCRSystem.
"""

import os
import argparse
import time
import cv2
from src.core.ocr_system import OCRSystem
from src.utils import template_gen

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser(description="Sistema OCR CV Puro (Sin ML)")
    parser.add_argument("--image", help="Ruta de la imagen a procesar (Requerido para modo CLI)")
    parser.add_argument("--gui", action="store_true", help="Iniciar interfaz gráfica web")
    parser.add_argument("--extract-images", action="store_true", help="Extraer imágenes y diagramas del documento")
    parser.add_argument("--output_dir", default="output", help="Carpeta de salida")
    parser.add_argument("--templates_dir", default="Dataset", help="Carpeta de plantillas de referencia")
    parser.add_argument("--generate_templates", action="store_true", help="Generar plantillas básicas antes de procesar")
    parser.add_argument("--debug", action="store_true", help="Habilitar modo debug (guardar caracteres)")
    
    args = parser.parse_args()

    # Modo GUI
    if args.gui:
        print("[INFO] Iniciando modo GUI...")
        try:
            from src.gui.gui import app
            app.run(host="127.0.0.1", port=5000)
        except Exception as e:
            print(f"[ERROR] No se pudo iniciar la GUI: {e}")
        return

    # Modo CLI: Validar imagen
    if not args.image:
        print("[ERROR] Debes especificar --image o --gui")
        parser.print_help()
        return

    # 0. Preparación
    if args.debug:
        print("[INFO] Modo DEBUG activado. Limpiando carpeta debug_chars...")
        import shutil
        if os.path.exists("debug_chars"):
            try:
                shutil.rmtree("debug_chars")
            except Exception as e:
                print(f"[WARN] No se pudo limpiar debug_chars: {e}")
        
        if not os.path.exists("debug_chars"):
            os.makedirs("debug_chars")

    if args.generate_templates or not os.path.exists(args.templates_dir):
        print("[INFO] Generando/Verificando plantillas...")
        template_gen.generate_templates(args.templates_dir)

    ensure_dir(args.output_dir)
    
    # Inicializar Sistema OCR
    ocr = OCRSystem(templates_dir=args.templates_dir, debug=args.debug)
    
    print(f"[INFO] Procesando: {args.image}")
    start_time = time.time()
    
    try:
        # Detectar tipo de texto (heurística simple o parámetro)
        text_type = "digital" # Default
        if "manuscrito" in args.image.lower():
            text_type = "manuscrito"

        result = ocr.process_image(args.image, text_type=text_type, extract_images=args.extract_images)
    except Exception as e:
        print(f"[ERROR] Fallo en el procesamiento: {e}")
        return

    elapsed = time.time() - start_time
    
    # Guardar Resultados
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # El sistema OCR ya guarda resultados en su propia estructura de carpetas (run_YYYYMMDD...)
    # pero mantenemos la lógica de impresión de resumen aquí.
    
    print(f"[DONE] Procesamiento completado en {elapsed:.2f}s")
    print(f"[INFO] Resultados guardados en: {result['output_dir']}")
    print(f"[INFO] Texto extraído (primeras lineas):")
    print(result["full_text"][:200] + "...")
    
    if args.extract_images:
        print(f"[INFO] Imágenes extraídas: {result['extracted_images_count']}")

if __name__ == "__main__":
    main()
