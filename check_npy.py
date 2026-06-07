import numpy as np

data = np.load("data/preprocessed_image.npy") #The variable image is member 2's FFT input.

print("Shape:", data.shape)
print("Data Type:", data.dtype)
print("Minimum:", data.min())
print("Maximum:", data.max())