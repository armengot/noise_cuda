import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('img/lena_hires.jpg')

# Obtener el tamaño de la imagen
height, width = image.shape[:2]

# Calcular el número de píxeles para aplicar el ruido
num_pixels = int(height * width * 0.15)

# Generar índices aleatorios para los píxeles a modificar
random_indices = np.random.choice(range(height * width), size=num_pixels, replace=False)

# Aplicar ruido a los píxeles seleccionados
image_flattened = image.reshape(-1, 3)
image_flattened[random_indices] = np.random.randint(0, 256, size=(num_pixels, 3))

# Volver a darle forma a la imagen
image_noisy = image_flattened.reshape(height, width, 3)

# Guardar la imagen con ruido
cv2.imwrite('lena_hires_noisy.jpg', image_noisy)
