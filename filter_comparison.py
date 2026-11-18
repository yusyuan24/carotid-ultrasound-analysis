from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# Configuration
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

FRAME_NAME = "frame_0000.png"

# Original image
ORIGINAL_PATH = PROJECT_DIR / "aligned_frames" / FRAME_NAME

# Filter results
ROOT_DIR = PROJECT_DIR / "filter_results"


FILTERS = [
    ("Mean", "mean"),
    ("Median", "median"),
    ("Lee Speckle", "lee_speckle"),
    ("Butterworth D0.50 Order10", "butterworth_D0.50_order10"),
    ("Butterworth D0.15 Order2", "butterworth_D0.15_order2"),
    ("Gaussian K5 Sigma0.1", "gaussian_k5_sigma0.1"),
    ("Gaussian K5 Sigma1.0", "gaussian_k5_sigma1.0"),
]




OUTPUT_PATH = ROOT_DIR / "filter_comparison_frame_0000.png"

# ==================================================
# Load images using OpenCV
# ==================================================

image_paths = [ORIGINAL_PATH] + [
    ROOT_DIR / folder / FRAME_NAME
    for _, folder in FILTERS
]

row_names = [
    "Original\n(Aligned Frame)",
    "Mean\n(K = 3)",
    "Median\n(K = 3)",
    "Lee Speckle\n(K = 3)",
    "Butterworth\n(cutoff = 0.50, order = 10)",
    "Butterworth\n(cutoff = 0.15, order = 2)",
    "Gaussian\n(K = 5, σ = 0.1)",
    "Gaussian\n(K = 5, σ = 1.0)",
    
]

images = []

for path in image_paths:

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    # Convert BGR to RGB if color image
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    print(
        f"{path.parent.name}: "
        f"dtype={image.dtype}, "
        f"shape={image.shape}, "
        f"min={image.min()}, "
        f"max={image.max()}, "
        f"mean={image.mean():.2f}"
    )

    images.append(image)


# ==================================================
# Plot: 2 rows x 4 columns
# ==================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(12, 6),
    squeeze=False,
)

for idx, (name, image) in enumerate(zip(row_names, images)):

    row = idx // 4
    col = idx % 4

    ax = axes[row, col]

    # Determine display range
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

    ax.set_title(
        name,
        fontsize=10,
        fontweight="bold",
        pad=8,
        linespacing=1.3,
    )

# ==================================================
# Layout
# ==================================================

# fig.suptitle(
#     "Filter Comparison: Frame 0",
#     fontsize=15,
#     fontweight="bold",
#     y=2,
# )

# fig.subplots_adjust(
#     left=0.03,
#     right=0.98,
#     top=0.90,
#     bottom=0.9,
#     wspace=0.08,
#     hspace=0.3,
# )

# ==================================================
# Save
# ==================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print(f"\nComparison saved to: {OUTPUT_PATH}")
