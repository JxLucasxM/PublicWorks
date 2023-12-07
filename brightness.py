import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from matplotlib.image import imread

def change_brightness(image, factor):
    # Step 1: Perform 2D Fourier Transform
    fourier_transform = fft2(image)

    # Step 2: Shift zero frequency components to the center
    fourier_transform_shifted = fftshift(fourier_transform)

    # Step 3: Scale the magnitude spectrum to change brightness
    magnitude_spectrum = np.abs(fourier_transform_shifted)
    magnitude_spectrum_scaled = magnitude_spectrum * factor

    # Step 4: Inverse Fourier Transform to reconstruct the image
    fourier_transform_scaled = fourier_transform_shifted * (magnitude_spectrum_scaled / magnitude_spectrum)
    image_scaled = np.abs(ifft2(ifftshift(fourier_transform_scaled)))

    return image_scaled

# Load your image or create a sample image for illustration
image_path = "c:/Users/jlucasmartinezwork/OneDrive/Desktop/Math450/Project/Images/pigeon.jpg"  # Change this to the actual path
image = imread(image_path)

brightness_factor = 0.01 # Adjust this factor for brightness control

# Change brightness using the defined function
brightened_image = change_brightness(image, brightness_factor)

# Visualize the results
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2), plt.imshow(brightened_image, cmap='gray')
plt.title('Brightened Image')

plt.show()