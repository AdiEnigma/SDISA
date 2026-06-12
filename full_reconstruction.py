import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv

# Load original preprocessed image
original = np.load('data/preprocessed_image.npy')

# Load filtered spectrum (for comparison)
filtered_spectrum = np.load('data/filtered_spectrum.npy')

# --- Reconstruct from ORIGINAL (exact) ---
fft_original = np.fft.fft2(original)
reconstructed_original = np.fft.ifft2(fft_original)
reconstructed_original = np.abs(reconstructed_original)  # Get magnitude

# --- Reconstruct from FILTERED spectrum ---
img_ishift = np.fft.ifftshift(filtered_spectrum)
img_filtered = np.fft.ifft2(img_ishift)
img_filtered = np.abs(img_filtered)  # Get magnitude

# Display comparison
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(original, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(132)
plt.imshow(reconstructed_original, cmap='gray')
plt.title('Reconstructed from FFT')
plt.xticks([]), plt.yticks([])

plt.subplot(133)
plt.imshow(img_filtered, cmap='gray')
plt.title('Reconstructed from Filtered')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

# Check if they're nearly identical
print(f"Original shape: {original.shape}")
print(f"Reconstructed shape: {reconstructed_original.shape}")
print(f"Max difference: {np.max(np.abs(original - reconstructed_original))}")