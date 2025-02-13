import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models

def url_to_base64(url):
    """Descarga una imagen desde la URL y la convierte a base64."""
    response = requests.get(url)
    return base64.b64encode(response.content).decode('utf-8')

def base64_to_image(b64_string):
    """Convierte una imagen en base64 a un objeto PIL."""
    img_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(img_data))

def preprocess_image(img, target_size=(224, 224)):
    """Preprocesa la imagen para el modelo de ResNet (o VGG)."""
    img = img.resize(target_size)  # Redimensionar la imagen
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir la dimensión del lote
    img_array = preprocess_input(img_array)  # Preprocesar para el modelo de ResNet
    return img_array

def extract_features(model, img_array):
    """Extrae las características de la imagen usando un modelo preentrenado."""
    return model.predict(img_array)

def cosine_similarity(features1, features2):
    """Calcula la similitud coseno entre dos conjuntos de características."""
    dot_product = np.dot(features1, features2.T)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    return dot_product / (norm1 * norm2)

def compare_images(img1_url, img2_url):
    """Compara dos imágenes utilizando un modelo de IA preentrenado y calcula la similitud en porcentaje."""
    # Cargar el modelo preentrenado (ResNet50)
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # Convertir imágenes a base64
    img1_b64 = url_to_base64(img1_url)
    img2_b64 = url_to_base64(img2_url)

    # Convertir base64 a imágenes PIL
    img1 = base64_to_image(img1_b64)
    img2 = base64_to_image(img2_b64)

    # Preprocesar las imágenes para el modelo
    img1_array = preprocess_image(img1)
    img2_array = preprocess_image(img2)

    # Extraer las características de las imágenes
    features1 = extract_features(model, img1_array)
    features2 = extract_features(model, img2_array)

    # Calcular la similitud coseno entre las características
    similarity_score = cosine_similarity(features1, features2)

    # Convertir la similitud en porcentaje
    similarity_percentage = similarity_score * 100

    return similarity_percentage

# URLs de ejemplo
img1_url = "https://medias.jeanpaulgaultier.com/cdn-cgi/image/width=570,quality=90,format=avif/medias/sys_master/images/hc7/hab/9855534170142/le-beau-eau-de-toilette-ingredients-jean-paul-gaultier.jpeg/le-beau-eau-de-toilette-ingredients-jean-paul-gaultier.jpeg.jpg"
#img2_url = "https://medias.jeanpaulgaultier.com/cdn-cgi/image/width=570,quality=90,format=avif/medias/sys_master/images/hc7/hab/9855534170142/le-beau-eau-de-toilette-ingredients-jean-paul-gaultier.jpeg/le-beau-eau-de-toilette-ingredients-jean-paul-gaultier.jpeg.jpg"
img2_url = "https://perfumeheadquarters.com/cdn/shop/files/jean-paul-gaultier-le-beau-fragrance-42-ozeau-de-toilettefragrance8435415017206-262370.jpg?v=1727092145&width=990"

similarity_percentage = compare_images(img1_url, img2_url)

print(f"🔹 **Similitud Final con IA (ResNet)**: {similarity_percentage:.2f}% 🔹")

