import cv2
import numpy as np

image = cv2.imread('img/lena_hires.jpg')
height, width = image.shape[:2]
npix = int(height * width * 0.15)
random_indices = np.random.choice(range(height * width), size=npix, replace=False)
image_flattened = image.reshape(-1, 3)
image_flattened[random_indices] = np.random.randint(0, 256, size=(npix, 3))
image_noisy = image_flattened.reshape(height, width, 3)

cv2.imwrite('lena_hires_noisy.jpg', image_noisy)
