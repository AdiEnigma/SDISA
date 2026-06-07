import numpy as np
import matplotlib.pyplot as plt
import os

# ── Load preprocessed image from Member 1 ──────────────────────────────
preprocessed_array = np.load('data/preprocessed_image.npy')
print("Loaded array shape:", preprocessed_array.shape)
print("Loaded array dtype:", preprocessed_array.dtype)

# ── Compute 2D FFT ─────────────────────────────────────────────────────
fft_output = np.fft.fft2(preprocessed_array)

# shift low frequencies to center
fft_shifted = np.fft.fftshift(fft_output)

# compute magnitude with log scaling
magnitude_log = np.log(1 + np.abs(fft_shifted))

# ── Display and Save Spectrum ───────────────────────────────────────────
os.makedirs('output', exist_ok=True)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(preprocessed_array, cmap='gray')
plt.title("Preprocessed Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_log, cmap='inferno')
plt.title("FFT Magnitude Spectrum")
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.savefig('output/spectrum_result.png', dpi=150, bbox_inches='tight')
plt.show()

print("Spectrum saved to output/spectrum_result.png")

# ── Save complex FFT output for Member 3 ───────────────────────────────
np.save('data/fft_shifted.npy', fft_shifted)
print("FFT data saved to data/fft_shifted.npy for Member 3")
print("FFT output shape:", fft_shifted.shape)
print("FFT output dtype:", fft_shifted.dtype)