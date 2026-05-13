import cv2
import numpy as np

# Fijar semilla para consistencia
np.random.seed(42)
cv2.setRNGSeed(42)


def to_grayscale(image):
    """Convierte la imagen a escala de grises."""
    # Convertir de BGR a GRIS (un solo canal)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image, width, height):
    """Redimensiona la imagen al tamaño indicado."""
    # Usar INTER_AREA para reducir o aumentar
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def apply_blur(image, kernel_size=5):
    """Aplica filtro Gaussiano de suavizado."""
    # Asegurar que kernel_size sea impar (si es par, se suma 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Aplicar Gaussian Blur
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """Ajusta brillo (beta) y contraste (alpha)."""
    # alpha: contraste (1.0 = sin cambios)
    # beta: brillo (0 = sin cambios)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_threshold(image, thresh_value=127):
    """Aplica umbralización binaria."""
    # Verificar si la imagen tiene múltiples canales
    if len(image.shape) == 3:
        # Si tiene 3 canales, convertir a grises primero
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Aplicar umbralización binaria
    _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    return binary


def detect_edges(image, low=50, high=150):
    """Detecta bordes con Canny."""
    # Convertir a grises si es necesario
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Aplicar detección de bordes Canny
    edges = cv2.Canny(gray, low, high)
    return edges


def full_pipeline(image, target_width=224, target_height=224):
    """
    Pipeline completo de preprocesamiento.
    Orden: 1) Redimensionar → 2) Grises → 3) Blur → 4) Bordes
    """
    # Paso 1: Redimensionar
    resized = resize_image(image, target_width, target_height)
    
    # Paso 2: Convertir a escala de grises
    gray = to_grayscale(resized)
    
    # Paso 3: Aplicar blur (kernel=3 según especificación)
    blurred = apply_blur(gray, kernel_size=3)
    
    # Paso 4: Detectar bordes
    edges = detect_edges(blurred, low=50, high=150)
    
    return edges
