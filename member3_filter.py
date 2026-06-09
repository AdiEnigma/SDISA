import numpy as np
import matplotlib.pyplot as plt
import os


# =====================================================
# SPECTRUM ANALYSIS FUNCTIONS
# =====================================================

def calculate_spectral_energy(spectrum):
    """
    Calculates total spectral energy.
    """
    return np.sum(np.abs(spectrum) ** 2)


def energy_retention(original_spectrum, filtered_spectrum):
    """
    Calculates percentage of energy retained after filtering.
    """
    original_energy = calculate_spectral_energy(original_spectrum)
    filtered_energy = calculate_spectral_energy(filtered_spectrum)

    if original_energy == 0:
        return 0

    return (filtered_energy / original_energy) * 100


def analyze_spectrum(spectrum):
    """
    Returns basic spectrum analysis metrics.
    """
    magnitude = np.abs(spectrum)

    return {
        "Total Energy": calculate_spectral_energy(spectrum),
        "Maximum Magnitude": np.max(magnitude),
        "Mean Magnitude": np.mean(magnitude),
        "Minimum Magnitude": np.min(magnitude)
    }


# =====================================================
# DISTANCE MATRIX
# =====================================================

def get_distance_matrix(shape):
    """
    Calculates Euclidean distance from center.
    """
    rows, cols = shape

    center_row = rows // 2
    center_col = cols // 2

    u, v = np.ogrid[:rows, :cols]

    D = np.sqrt(
        (u - center_row) ** 2 +
        (v - center_col) ** 2
    )

    return D


# =====================================================
# FILTERS
# =====================================================

def apply_low_pass_filter(complex_spectrum, D0):
    """
    Low-pass filter.
    Keeps frequencies inside radius D0.
    """
    D = get_distance_matrix(complex_spectrum.shape)

    mask = D <= D0

    filtered_spectrum = complex_spectrum * mask

    return filtered_spectrum, mask


def apply_high_pass_filter(complex_spectrum, D0):
    """
    High-pass filter.
    Keeps frequencies outside radius D0.
    """
    D = get_distance_matrix(complex_spectrum.shape)

    mask = D > D0

    filtered_spectrum = complex_spectrum * mask

    return filtered_spectrum, mask


def apply_band_pass_filter(complex_spectrum, D_min, D_max):
    """
    Band-pass filter.
    Keeps frequencies between D_min and D_max.
    """
    D = get_distance_matrix(complex_spectrum.shape)

    mask = (D >= D_min) & (D <= D_max)

    filtered_spectrum = complex_spectrum * mask

    return filtered_spectrum, mask


# =====================================================
# VISUALIZATION
# =====================================================

def display_spectrum(spectrum, title="Spectrum"):
    """
    Displays log-scaled magnitude spectrum.
    """
    magnitude = np.log1p(np.abs(spectrum))

    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def display_mask(mask, title="Mask"):
    """
    Displays filter mask.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


# =====================================================
# SAVE FILTERED SPECTRUM
# =====================================================

def save_filtered_spectrum(
        filtered_spectrum,
        output_path="data/filtered_spectrum.npy"
):
    """
    Saves filtered spectrum for Member 4.
    """

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    np.save(output_path, filtered_spectrum)

    print(f"\nFiltered spectrum saved to: {output_path}")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    INPUT_FILE = "data/fft_shifted.npy"

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        exit()

    # ---------------------------------------------
    # Load FFT spectrum from Member 2
    # ---------------------------------------------

    spectrum = np.load(INPUT_FILE)

    print("\n========== ORIGINAL SPECTRUM ==========")

    analysis = analyze_spectrum(spectrum)

    for key, value in analysis.items():
        print(f"{key}: {value}")

    display_spectrum(
        spectrum,
        "Original FFT Spectrum"
    )

    # ---------------------------------------------
    # Filter Selection
    # ---------------------------------------------

    print("\nSelect Filter:")
    print("1. Low Pass Filter")
    print("2. High Pass Filter")
    print("3. Band Pass Filter")

    choice = input("Enter choice (1/2/3): ")

    if choice == "1":

        filtered_spectrum, mask = apply_low_pass_filter(
            spectrum,
            D0=30
        )

        filter_name = "Low Pass"

    elif choice == "2":

        filtered_spectrum, mask = apply_high_pass_filter(
            spectrum,
            D0=30
        )

        filter_name = "High Pass"

    elif choice == "3":

        filtered_spectrum, mask = apply_band_pass_filter(
            spectrum,
            D_min=20,
            D_max=60
        )

        filter_name = "Band Pass"

    else:
        raise ValueError("Invalid filter choice.")

    # ---------------------------------------------
    # Results
    # ---------------------------------------------

    print(f"\n========== {filter_name.upper()} FILTER ==========")

    retained = energy_retention(
        spectrum,
        filtered_spectrum
    )

    print(f"Energy Retained: {retained:.2f}%")

    display_mask(
        mask,
        f"{filter_name} Mask"
    )

    display_spectrum(
        filtered_spectrum,
        f"{filter_name} Filtered Spectrum"
    )

    # ---------------------------------------------
    # Save individual output
    # ---------------------------------------------

    output_filename = (
        f"data/{filter_name.lower().replace(' ', '_')}_spectrum.npy"
    )

    np.save(
        output_filename,
        filtered_spectrum
    )

    print(f"Saved: {output_filename}")

    # ---------------------------------------------
    # Save standard output for Member 4
    # ---------------------------------------------

    save_filtered_spectrum(
        filtered_spectrum,
        "data/filtered_spectrum.npy"
    )

    print("\nMember 3 module executed successfully.")
