import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv

#load filtered spectrum
img = cv.imread('output/preprocessed_image_normalized.png', cv.IMREAD_GRAYSCALE)
low_filtered_spectrum = np.load('data/low_pass_spectrum.npy')
print("shape:", low_filtered_spectrum.shape)
print("dtype:", low_filtered_spectrum.dtype)

high_filtered_spectrum = np.load('data/high_pass_spectrum.npy')
print("shape:", high_filtered_spectrum.shape)
print("dtype:", high_filtered_spectrum.dtype)

band_filtered_spectrum = np.load('data/band_pass_spectrum.npy')
print("shape:", band_filtered_spectrum.shape)
print("dtype:", band_filtered_spectrum.dtype)

img_ishift_low = np.fft.ifftshift(low_filtered_spectrum)
img_back_low = np.fft.ifft2(img_ishift_low)
img_back_low = np.real(img_back_low)

img_ishift_high = np.fft.ifftshift(high_filtered_spectrum)
img_back_high = np.fft.ifft2(img_ishift_high)
img_back_high = np.real(img_back_high)

img_ishift_band = np.fft.ifftshift(band_filtered_spectrum)
img_back_band = np.fft.ifft2(img_ishift_band)
img_back_band = np.real(img_back_band)

plt.subplot(141)
plt.imshow(img, cmap='summer')
plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(142)
plt.imshow(img_back_low, cmap='gray')
plt.title('Low-Pass Filtered Image'),plt.xticks([]), plt.yticks([])
plt.subplot(143)
plt.imshow(img_back_band, cmap='gray')
plt.title('Band-Pass Filtered Image'),plt.xticks([]), plt.yticks([])
plt.subplot(144)
plt.imshow(img_back_high, cmap='gray')
plt.title('High-Pass Filtered Image'),plt.xticks([]), plt.yticks([])



plt.tight_layout()

plt.show()
