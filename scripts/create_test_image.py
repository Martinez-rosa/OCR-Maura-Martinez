from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image():
    # Create white image
    img = Image.new('RGB', (800, 600), color='white')
    d = ImageDraw.Draw(img)
    
    # Try to load a font
    try:
        # Try direct path first (Windows)
        font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf")
        font = ImageFont.truetype(font_path, 40)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            print("Warning: Arial font not found. Using default (tiny) font.")
            font = ImageFont.load_default()
    
    # Add text
    d.text((50, 50), "HOLA MUNDO", fill='black', font=font)
    d.text((50, 150), "PRUEBA DE DEBUG", fill='black', font=font)
    d.text((50, 250), "1234567890", fill='black', font=font)
    
    # Save
    img.save("test_debug_image.png")
    print("Imagen de prueba creada: test_debug_image.png")

if __name__ == "__main__":
    create_test_image()
