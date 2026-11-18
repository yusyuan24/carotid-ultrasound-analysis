
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# Configuration
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

FRAME_NAME = "frame_0000.png"

# Original filter images
FILTER_ROOT = PROJECT_DIR / "filter_results"

# Contrast enhancement results
CONTRAST_ROOT = PROJECT_DIR / "contrast_results"

# Output directory
OUTPUT_DIR = PROJECT_DIR / "contrast_results"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Seven filter configurations
FILTERS = [
    ("Mean (K = 3)", "mean"),
    ("Median (K = 3)", "median"),
    ("Lee Speckle (K = 3)", "lee_speckle"),
    ("Butterworth (cutoff = 0.50, order = 10)", "butterworth_D0.50_order10"),
    ("Butterworth (cutoff = 0.15, order = 2)", "butterworth_D0.15_order2"),
    ("Gaussian (K = 5, σ = 0.1)", "gaussian_k5_sigma0.1"),
    ("Gaussian (K = 5, σ = 1.0)", "gaussian_k5_sigma1.0"),
]

row_names = [
    "Mean (K = 3)",
    "Median (K = 3)",
    "Lee Speckle (K = 3)",
    "Butterworth\n(cutoff = 0.50, order = 10)",
    "Butterworth\n(cutoff = 0.15, order = 2)",
    "Gaussian\n(K = 5, σ = 0.1)",
    "Gaussian\n(K = 5, σ = 1.0)",
]
# Contrast enhancement methods
CONTRAST_METHODS = [
    ("CLAHE", "adapthisteq_clahe"),
    ("Histogram Equalization", "histogram_equalization"),
    ("Intensity Adjustment", "imadjust"),
]

# ==================================================
# Image loading function
# ==================================================

def load_image(path):

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    # Convert BGR to RGB if necessary
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    return image


# ==================================================
# Image display function
# ==================================================

def display_image(ax, image):

    if image.dtype == np.uint8:
        vmin, vmax = 0, 255

    elif image.dtype == np.uint16:
        vmin, vmax = 0, 65535

    elif np.issubdtype(image.dtype, np.floating):
        vmin, vmax = 0, 1

    else:
        vmin, vmax = float(image.min()), float(image.max())

    ax.imshow(
        image,
        cmap="gray" if image.ndim == 2 else None,
        vmin=vmin,
        vmax=vmax,
    )

    ax.axis("off")


# ==================================================
# Generate seven comparison figures
# ==================================================

for filter_idx, (filter_name, filter_folder) in enumerate(FILTERS):

    print(f"\nProcessing: {filter_name}")

    # Original filter image
    original_path = (
        FILTER_ROOT / filter_folder / FRAME_NAME
    )

    original_image = load_image(original_path)

    # Create image list
    images = [
        (f"Filter: {filter_name}", original_image),
    ]

    # Load three contrast enhancement results
    for method_name, method_folder in CONTRAST_METHODS:

        contrast_path = (
            CONTRAST_ROOT
            / filter_folder
            / method_folder
            / FRAME_NAME
        )

        contrast_image = load_image(contrast_path)

        images.append((method_name, contrast_image))
    

    # ==================================================
    # Plot: 1 row x 4 columns
    # ==================================================

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), squeeze=False)

    for idx, (title, image) in enumerate(images):
        ax = axes[0, idx]
        display_image(ax, image)

        # 第一張圖片標題與主標題分開
        if idx == 0:
            # ax.set_title(row_names[filter_idx], fontsize=14, fontweight="bold", pad=8)
            ax.set_title("Filtered Image", fontsize=14, fontweight="bold", pad=8)
        else:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=8)

    fig.suptitle(
        f"Contrast Enhancement Comparison: {filter_name}",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.85,
        bottom=0.03,
        wspace=0.08
    )

    output_path = OUTPUT_DIR / f"{filter_name}_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)