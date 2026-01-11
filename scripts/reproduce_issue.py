from PIL import Image, ImageDraw, ImageFont
import os

def create_reproduction_image():
    # Crear imagen blanca
    width = 800
    height = 600
    img = Image.new('RGB', (width, height), color='white')
    d = ImageDraw.Draw(img)
    
    # Intentar cargar Arial
    try:
        font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf")
        # Tamaño aproximado al de la captura (parece ser cuerpo de texto normal)
        font_header = ImageFont.truetype(font_path, 24) 
        font_body = ImageFont.truetype(font_path, 18)
    except IOError:
        font_header = ImageFont.load_default()
        font_body = ImageFont.load_default()
    
    # Color texto: Gris oscuro/Negro (simular captura real)
    text_color = (40, 40, 40)
    
    # Dibujar texto
    # Header
    d.text((50, 50), "Temario del Curso", fill=(0, 102, 204), font=font_header) # Azulito como en la captura
    
    # Body items
    y_start = 100
    line_height = 40
    
    items = [
        "Modulo 1: Introduccion a la automatizacion inmobiliaria",
        "Modulo 2: Automatizacion de comunicacion y email",
        "Modulo 3: Automatizacion documental",
        "Modulo 4: Flujos completos de procesos inmobiliarios",
        "Modulo 5: Taller practico final"
    ]
    
    for i, item in enumerate(items):
        # Simular bullet point
        y = y_start + i * line_height
        d.text((50, y), "•", fill='black', font=font_body)
        d.text((70, y), item, fill='black', font=font_body)
    
    # Save
    filename = "reproduce_issue.png"
    img.save(filename)
    print(f"Imagen de reproducción creada: {filename}")

if __name__ == "__main__":
    create_reproduction_image()
