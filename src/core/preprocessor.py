"""
Módulo de preprocesamiento de imágenes para OCR
Funciones para normalizar y limpiar caracteres manuscritos
"""

import cv2
import numpy as np
from skimage import morphology

class ImagePreprocessor:
    """
    Preprocesa imágenes de caracteres para mejorar reconocimiento.
    Pipeline:
      binarize → remove_noise → crop_to_content → (normalize_thickness?) →
      resize_with_padding → center_mass.
    target_size define el tamaño final (por defecto 32x32) alineado con los
    reconocedores y comparadores del sistema.
    """
    
    def __init__(self, target_size=(32, 32)):
        """
        Args:
            target_size (tuple): Tamaño objetivo (ancho, alto)
        """
        self.target_size = target_size
    
    def binarize(self, img):
        """
        Convierte imagen a binario blanco/negro puro
        
        Args:
            img: Imagen en escala de grises
        Returns:
            Imagen binarizada (0=negro, 255=blanco)
        """
        # Si la imagen ya es a color, convertir a gris
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarización con umbral de Otsu (automático)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Asegurar que el fondo sea blanco y el texto negro
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def remove_noise(self, img):
        """
        Elimina ruido pequeño usando morfología
        
        Args:
            img: Imagen binaria
        Returns:
            Imagen sin ruido
        """
        # Apertura morfológica: elimina puntos blancos pequeños
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Cierre morfológico: rellena huecos pequeños
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return img
    
    def crop_to_content(self, img):
        """
        Recorta la imagen al contenido (elimina bordes blancos)
        
        Args:
            img: Imagen binaria
        Returns:
            Imagen recortada
        """
        # Encontrar píxeles negros (contenido)
        coords = cv2.findNonZero(cv2.bitwise_not(img))
        
        if coords is None:
            # Si no hay contenido, devolver imagen original
            return img
        
        # Obtener bounding box
        x, y, w, h = cv2.boundingRect(coords)
        
        # Añadir pequeño margen (5%)
        margin = int(min(w, h) * 0.05)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        
        # Recortar
        cropped = img[y:y+h, x:x+w]
        
        return cropped
    
    def resize_with_padding(self, img):
        """
        Redimensiona manteniendo aspect ratio y añade padding
        
        Args:
            img: Imagen a redimensionar
        Returns:
            Imagen redimensionada con padding
        """
        h, w = img.shape
        if h == 0 or w == 0:
            return np.ones(self.target_size, dtype=np.uint8) * 255

        target_h, target_w = self.target_size
        
        # Calcular factor de escala manteniendo aspect ratio
        scale = min(target_w / w, target_h / h)
        
        # Nuevo tamaño
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Asegurar dimensiones mínimas de 1px para evitar errores en reescalado
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        # Redimensionar
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Crear imagen con padding blanco
        padded = np.ones(self.target_size, dtype=np.uint8) * 255
        
        # Centrar imagen redimensionada
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def center_mass(self, img):
        """
        Centra el carácter por centro de masa
        
        Args:
            img: Imagen binaria
        Returns:
            Imagen centrada
        """
        # Calcular momentos
        moments = cv2.moments(cv2.bitwise_not(img))
        
        if moments['m00'] == 0:
            return img
        
        # Centro de masa
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # Centro de la imagen
        h, w = img.shape
        center_x, center_y = w // 2, h // 2
        
        # Desplazamiento necesario
        shift_x = center_x - cx
        shift_y = center_y - cy
        
        # Matriz de traslación
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # Aplicar traslación
        centered = cv2.warpAffine(img, M, (w, h), 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=255)
        
        return centered
    
    def normalize_thickness(self, img):
        """
        Normaliza el grosor del trazo (adelgazamiento)
        
        Args:
            img: Imagen binaria
        Returns:
            Imagen con trazo normalizado
        """
        # Invertir para skeletonize (necesita True=objeto)
        inverted = img < 128
        
        # Adelgazamiento morfológico
        skeleton = morphology.skeletonize(inverted)
        
        # Dilatar ligeramente para dar grosor uniforme
        skeleton = morphology.binary_dilation(skeleton, morphology.square(2))
        
        # Convertir de vuelta a uint8
        result = (~skeleton * 255).astype(np.uint8)
        
        return result
    
    def preprocess(self, img, normalize_thickness=False):
        """
        Pipeline completo de preprocesamiento
        
        Args:
            img: Imagen de entrada (puede ser color o gris)
            normalize_thickness: Si True, normaliza grosor de trazo
        Returns:
            Imagen preprocesada lista para reconocimiento
        """
        # 1. Binarizar
        img = self.binarize(img)
        
        # 2. Eliminar ruido
        img = self.remove_noise(img)
        
        # 3. Recortar al contenido
        img = self.crop_to_content(img)
        
        # 4. Normalizar grosor (opcional, puede ayudar con manuscrita)
        if normalize_thickness:
            try:
                img = self.normalize_thickness(img)
            except:
                pass  # Si falla, continuar sin normalizar
        
        # 5. Redimensionar con padding
        img = self.resize_with_padding(img)
        
        # 6. Centrar por centro de masa
        img = self.center_mass(img)
        
        return img


# Función de conveniencia
def preprocess_image(img_path, target_size=(32, 32), normalize_thickness=False):
    """
    Función simple para preprocesar una imagen desde archivo
    
    Args:
        img_path: Ruta a la imagen
        target_size: Tamaño objetivo
        normalize_thickness: Normalizar grosor de trazo
    Returns:
        Imagen preprocesada
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")
    
    preprocessor = ImagePreprocessor(target_size)
    return preprocessor.preprocess(img, normalize_thickness)


if __name__ == '__main__':
    # Prueba del preprocesador
    import matplotlib.pyplot as plt
    
    # Cargar imagen de ejemplo
    test_img = cv2.imread('test_char.png', cv2.IMREAD_GRAYSCALE)
    
    if test_img is not None:
        preprocessor = ImagePreprocessor()
        
        # Preprocesar
        processed = preprocessor.preprocess(test_img)
        
        # Mostrar resultado
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(test_img, cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(processed, cmap='gray')
        plt.title('Preprocesada')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Preprocesamiento exitoso")
    else:
        print("⚠ No se encontró imagen de prueba")
