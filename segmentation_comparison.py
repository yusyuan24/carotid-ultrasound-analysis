"""Compare six lumen-segmentation methods on one representative frame.

Keep the preprocessing input fixed while comparing segmentation methods.
The configuration below records the user's current best observed setting.
This script is intentionally not run automatically.
"""
from pathlib import Path
import csv

import cv2
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
FILTER_METHOD = "gaussian_k5_sigma1.0"
CONTRAST_METHOD = "histogram_equalization"
CONTRAST = "Histogram Equalization"


FRAME_INDEX = 0
INPUT_DIR = ROOT / "contrast_results" / FILTER_METHOD / CONTRAST_METHOD
OUTPUT_DIR = ROOT / "segmentation_comparison_results" / (
    f"{FILTER_METHOD}_{CONTRAST_METHOD}_frame{FRAME_INDEX}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Coordinates are relative to the user's 200 x 200 crop.
VESSEL_CENTER_XY = (105, 100)
ROI_XYXY = (35, 165, 45, 175)  # x0, x1, y0, y1
PRIOR_RADIUS = 55
MM_PER_PIXEL = 0.062397
REGION_GROW_TOLERANCE = 80
FCM_CLUSTERS = 4
FCM_FUZZINESS = 2.0
FCM_MAX_ITERATIONS = 100
FCM_SOFT_MEMBERSHIP_THRESHOLD = 0.35
HYBRID_CORE_ERODE_ITERATIONS = 1

FILTERS = [
    ("Mean (K = 3)", "mean"),
    ("Median (K = 3)", "median"),
    ("Lee Speckle (K = 3)", "lee_speckle"),
    ("Butterworth (cutoff = 0.50, order = 10)", "butterworth_D0.50_order10"),
    ("Butterworth (cutoff = 0.15, order = 2)", "butterworth_D0.15_order2"),
    ("Gaussian (K = 5, σ = 0.1)", "gaussian_k5_sigma0.1"),
    ("Gaussian (K = 5, σ = 1.0)", "gaussian_k5_sigma1.0"),
]
FILTER = next((name for name, method in FILTERS if method == FILTER_METHOD),None)
     


def circular_prior(shape, center_xy, radius):
    cx, cy = center_xy
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def select_component(mask, center_xy):
    """Keep the plausible dark region closest to the expected lumen center."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cx, cy = center_xy
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        px, py = centroids[label]
        distance = float(np.hypot(px - cx, py - cy))
        if 300 <= area <= 6000 and distance <= 28:
            candidates.append((distance + abs(area - 2500) / 1000, label))
    if not candidates:
        return binary * 255
    _, label = min(candidates, key=lambda item: item[0])
    return (labels == label).astype(np.uint8) * 255


def otsu_segmentation(roi, center_local):
    threshold, mask = cv2.threshold(
        roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    mask[~circular_prior(roi.shape, center_local, PRIOR_RADIUS)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return select_component(mask, center_local), float(threshold)


def region_growing(roi, center_local):
    """Four-neighbor seed growth using intensity similarity to the seed."""
    from collections import deque

    height, width = roi.shape
    seed_x, seed_y = map(int, center_local)
    seed_value = int(roi[seed_y, seed_x])
    prior = circular_prior(roi.shape, center_local, PRIOR_RADIUS)
    visited = np.zeros((height, width), dtype=bool)
    mask = np.zeros((height, width), dtype=np.uint8)
    queue = deque([(seed_x, seed_y)])
    visited[seed_y, seed_x] = True
    while queue:
        x, y = queue.popleft()
        if abs(int(roi[y, x]) - seed_value) > REGION_GROW_TOLERANCE:
            continue
        if not prior[y, x]:
            continue
        mask[y, x] = 255
        for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return select_component(mask, center_local), float("nan")


def fcm_membership_maps(roi):
    """Return per-pixel FCM memberships and learned gray-level centers."""
    values = roi.astype(np.float32).reshape(-1, 1)
    initial_percentiles = np.linspace(15, 85, FCM_CLUSTERS)
    centers = np.percentile(
        values, initial_percentiles
    ).astype(np.float32).reshape(-1)
    fuzzifier = FCM_FUZZINESS
    for _ in range(FCM_MAX_ITERATIONS):
        distances = np.abs(values - centers[None, :]) + 1e-6
        inv = distances ** (-2.0 / (fuzzifier - 1.0))
        memberships = inv / np.sum(inv, axis=1, keepdims=True)
        weights = memberships ** fuzzifier
        updated = np.sum(weights * values, axis=0) / (
            np.sum(weights, axis=0) + 1e-6
        )
        if np.max(np.abs(updated - centers)) < 0.01:
            centers = updated
            break
        centers = updated
    return memberships.reshape(*roi.shape, FCM_CLUSTERS), centers


def fuzzy_c_means(roi, center_local):
    """Hard-label FCM baseline using the darkest intensity cluster."""
    memberships, centers = fcm_membership_maps(roi)
    labels = np.argmax(memberships, axis=2)
    darkest_cluster = int(np.argmin(centers))
    mask = (labels == darkest_cluster).astype(np.uint8) * 255
    mask[~circular_prior(roi.shape, center_local, PRIOR_RADIUS)] = 0
    return select_component(mask, center_local), float("nan")


def region_growing_fcm_hybrid(roi, center_local):
    """Use region growing for a lumen core and soft FCM for its boundary."""
    growing_core, _ = region_growing(roi, center_local)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core = cv2.erode(
        growing_core, kernel, iterations=HYBRID_CORE_ERODE_ITERATIONS
    )

    memberships, centers = fcm_membership_maps(roi)
    darkest_cluster = int(np.argmin(centers))
    fuzzy_membership = memberships[:, :, darkest_cluster]
    candidate = (
        fuzzy_membership >= FCM_SOFT_MEMBERSHIP_THRESHOLD
    ).astype(np.uint8) * 255
    candidate[~circular_prior(roi.shape, center_local, PRIOR_RADIUS)] = 0
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    # Keep only the fuzzy component that overlaps the grown lumen core.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8), connectivity=8
    )
    candidates = []
    for label in range(1, count):
        component = labels == label
        overlap = int(np.count_nonzero(component & (core > 0)))
        if overlap == 0:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]
        distance = float(np.hypot(cx - center_local[0], cy - center_local[1]))
        candidates.append((overlap, -distance, -area, label))

    if candidates:
        # Favor a component supported by the core, then the one nearest the seed.
        label = max(candidates)[3]
        fuzzy_edge = (labels == label).astype(np.uint8) * 255
    else:
        fuzzy_edge = np.zeros_like(candidate)

    hybrid = cv2.bitwise_or(core, fuzzy_edge)
    hybrid[~circular_prior(roi.shape, center_local, PRIOR_RADIUS)] = 0
    return hybrid, float(FCM_SOFT_MEMBERSHIP_THRESHOLD)


def watershed_segmentation(roi, center_local):
    """Marker-controlled watershed initialized from an Otsu dark-region mask."""
    _, dark = cv2.threshold(
        roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    dark[~circular_prior(roi.shape, center_local, PRIOR_RADIUS)] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    selected = select_component(dark, center_local)

    distance = cv2.distanceTransform(selected, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0:
        return selected, float("nan")

    sure_foreground = (distance >= 0.45 * max_distance).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    sure_background = cv2.dilate(selected, kernel, iterations=2) > 0
    unknown = sure_background & (sure_foreground == 0)

    markers = np.zeros(roi.shape, dtype=np.int32)
    markers[~sure_background] = 1
    markers[sure_foreground > 0] = 2
    markers[unknown] = 0
    color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    result = (markers == 2).astype(np.uint8) * 255
    return select_component(result, center_local), float("nan")


def active_contour_segmentation(roi, center_local):
    """Snake/active contour initialized as a circle around the lumen."""
    try:
        from skimage.segmentation import active_contour
    except ImportError as exc:
        raise ImportError(
            "The active-contour method requires scikit-image. "
            "Install it with: python -m pip install scikit-image"
        ) from exc

    cx, cy = center_local
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    initial_snake = np.column_stack((
        cy + 0.85 * PRIOR_RADIUS * np.sin(theta),
        cx + 0.85 * PRIOR_RADIUS * np.cos(theta),
    ))
    snake = active_contour(
        roi.astype(np.float32) / 255.0,
        initial_snake,
        alpha=0.0015,
        beta=25.0,
        gamma=0.001,
        w_line=-0.5,
        w_edge=1.0,
        max_num_iter=300,
        convergence=0.1,
        boundary_condition="periodic",
    )
    polygon_xy = np.round(snake[:, ::-1]).astype(np.int32)
    mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_xy], 255)
    mask[~circular_prior(roi.shape, center_local, PRIOR_RADIUS)] = 0
    return select_component(mask, center_local), float("nan")


def vertical_diameter(mask):
    widths = []
    for row in range(mask.shape[0]):
        xs = np.flatnonzero(mask[row] > 0)
        if xs.size >= 3:
            widths.append(int(xs[-1] - xs[0] + 1))
    if len(widths) < 5:
        return float("nan")
    return float(np.median(widths))


def main():
    frame_path = INPUT_DIR / f"frame_{FRAME_INDEX:04d}.png"
    image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(
            f"Cannot read {frame_path}; check the filter/contrast selection and frame index."
        )
    if image.shape != (200, 200):
        raise ValueError(f"Expected a 200x200 crop, got {image.shape}.")

    x0, x1, y0, y1 = ROI_XYXY
    roi = image[y0:y1, x0:x1]
    center_x, center_y = VESSEL_CENTER_XY
    center_local = (center_x - x0, center_y - y0)

    methods = {
        "Otsu": otsu_segmentation,
        "Region growing": region_growing,
        "Fuzzy c-means": fuzzy_c_means,
        "Watershed": watershed_segmentation,
        "Active contour": active_contour_segmentation,
    }
    masks = {}
    threshold_values = {}
    for method, function in methods.items():
        mask_roi, threshold = function(roi, center_local)
        full_mask = np.zeros(image.shape, dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = mask_roi
        masks[method] = full_mask
        threshold_values[method] = threshold
        cv2.imwrite(
            str(OUTPUT_DIR / f"mask_{method.lower().replace(' ', '_')}.png"),
            full_mask,
        )

    # Arrange six logical panels as 2 rows x 3 columns. Each method panel
    # contains its overlay above its binary mask; the input occupies cell 1.
    method_panels = []
    for method, mask in masks.items():
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        overlay[mask > 0] = (
            0.55 * overlay[mask > 0] + 0.45 * np.array([0, 255, 255])
        ).astype(np.uint8)
        method_panels.append(
            (method, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), mask)
        )

    fig = plt.figure(figsize=(8, 11))
    grid = fig.add_gridspec(
        5, 3, height_ratios=[1.0, 1.0, 0.1, 1.0, 1.0],
        hspace=0.1, wspace=0.1,
    )
   
    # First logical row: original, method 1, method 2.
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Input", fontsize=12)
    ax.axis("off")
    for col, (title, overlay_panel, mask) in enumerate(method_panels[:2], start=1):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(overlay_panel)
        ax.set_title(title,fontsize=12)
        ax.axis("off")
        mask_ax = fig.add_subplot(grid[1, col])
        mask_ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
        # mask_ax.set_title(f"{title} mask")
        mask_ax.axis("off")

    # Grid row 2 is a blank spacer between the two logical rows.
    # Second logical row: methods 3, 4, and 5, with each mask underneath.
    for col, (title, overlay_panel, mask) in enumerate(method_panels[2:5]):
        ax = fig.add_subplot(grid[3, col])
        ax.imshow(overlay_panel)
        ax.set_title(title,fontsize=12)
        ax.axis("off")
        mask_ax = fig.add_subplot(grid[4, col])
        mask_ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
        # mask_ax.set_title(f"{title} mask")
        mask_ax.axis("off")

    fig.suptitle(
        f"Segmentation Methods on frame {FRAME_INDEX}\n"
        f"{FILTER} + {CONTRAST}", 
        fontsize=14,
        y=0.96,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=0.8)
    fig.savefig(OUTPUT_DIR / "segmentation_methods_comparison.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    with (OUTPUT_DIR / "segmentation_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "method", "frame", "method_parameter", "area_px",
            "vertical_diameter_px", "vertical_diameter_mm",
        ])
        for method, mask in masks.items():
            area = int(np.count_nonzero(mask))
            diameter_px = vertical_diameter(mask)
            writer.writerow([
                method, FRAME_INDEX, threshold_values[method], area,
                diameter_px,
                diameter_px * MM_PER_PIXEL if np.isfinite(diameter_px) else "",
            ])

    print(f"Comparison outputs saved in {OUTPUT_DIR}.")
    print("Inspect the masks before selecting a method for the full cine.")


if __name__ == "__main__":
    main()
