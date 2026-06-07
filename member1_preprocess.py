import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale,
    resizes it to 256x256, normalizes pixel values,
    and saves the processed array.
    """

    print("Loading image...")

    # Load image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("Image loaded successfully")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Converted to grayscale")

    # Resize image
    resized = cv2.resize(gray, (256, 256))

    print("Resized to 256 x 256")

    # Normalize pixel values (0-255 -> 0-1)
    normalized = resized.astype(np.float32) / 255.0

    print("Normalized between 0 and 1")

    # Create folders if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Save processed array for Member 2
    np.save(
        "data/preprocessed_image.npy",
        normalized
    )

    print("Saved preprocessed image")

    return img, gray, resized, normalized


# -----------------------------
# MAIN PROGRAM
# -----------------------------

if __name__ == "__main__":

    original, gray, resized, normalized = preprocess_image(
        "test_image.jpg"
    )

    print("\n------ IMAGE DETAILS ------")
    print("Original Shape   :", original.shape)
    print("Gray Shape       :", gray.shape)
    print("Resized Shape    :", resized.shape)
    print("Normalized Shape :", normalized.shape)

    # Display images
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(resized, cmap="gray")
    plt.title("Resized")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(normalized, cmap="gray")
    plt.title("Normalized")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        "output/preprocessing_result.png"
    )

    plt.show()

    print("\nPreprocessing completed successfully!")