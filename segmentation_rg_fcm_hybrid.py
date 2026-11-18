"""Hybrid lumen segmentation using two preprocessing pipelines.

Region Growing on Butterworth + CLAHE supplies the lumen core.
Fuzzy C-means on Gaussian + global histogram equalization supplies the edge.
The inputs are from the same frame and retain the same spatial coordinates.
This script is intentionally not run automatically.
"""
from pathlib import Path
import csv
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FRAME_INDEX = 0
REGION_INPUT_DIR = (
    ROOT / "contrast_results" / "butterworth_D0.50_order10"
    / "adapthisteq_clahe"
)

FCM_INPUT_DIR = (
    ROOT / "contrast_results" / "gaussian_k5_sigma1.0"
    / "histogram_equalization"
)
OUTPUT_DIR = ROOT / "segmentation_hybrid_results" / f"frame_{FRAME_INDEX:04d}"

# Current working spatial parameters; edit these if the crop/seed changes.

VESSEL_CENTER_XY = (103, 90)
ROI_XYXY = (35, 165, 45, 175)
PRIOR_RADIUS = 60
REGION_GROW_TOLERANCE = 100
FCM_CLUSTERS = 4
FCM_FUZZINESS = 2.0
FCM_MAX_ITERATIONS = 100
FCM_EDGE_MEMBERSHIP_THRESHOLD = 0.3
CORE_EROSION_KERNEL = 5
EDGE_REFINEMENT_KERNEL = 30
MM_PER_PIXEL = 0.062397


def read_frame(folder):
    path = folder / f"frame_{FRAME_INDEX:04d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    if image.shape != (200, 200):
        raise ValueError(f"Expected 200x200 input, got {image.shape} at {path}")
    return image


def circular_prior(shape, center_xy, radius):
    cx, cy = center_xy
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def choose_component(mask, center_xy, overlap_reference=None):
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cx, cy = center_xy
    candidates = []
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        px, py = centroids[label]
        distance = float(np.hypot(px - cx, py - cy))
        overlap = (
            int(np.count_nonzero(component & (overlap_reference > 0)))
            if overlap_reference is not None else 0
        )
        if area >= 20:
            candidates.append((overlap, -distance, -abs(area - 2500), label))
    if not candidates:
        return np.zeros_like(binary, dtype=np.uint8) * 255
    label = max(candidates)[3]
    return (labels == label).astype(np.uint8) * 255


def region_growing_core(roi, center_local):
    seed_x, seed_y = map(int, center_local)
    height, width = roi.shape
    seed_value = int(roi[seed_y, seed_x])
    prior = circular_prior(roi.shape, center_local, PRIOR_RADIUS)
    visited = np.zeros((height, width), dtype=bool)
    mask = np.zeros((height, width), dtype=np.uint8)
    queue = deque([(seed_x, seed_y)])
    visited[seed_y, seed_x] = True

    while queue:
        x, y = queue.popleft()
        if not prior[y, x]:
            continue
        if abs(int(roi[y, x]) - seed_value) > REGION_GROW_TOLERANCE:
            continue
        mask[y, x] = 255
        for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))

    mask = choose_component(mask, center_local)
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CORE_EROSION_KERNEL, CORE_EROSION_KERNEL)
    )
    core = cv2.erode(mask, erode_kernel, iterations=1)
    if not np.any(core):
        core = mask
    return core


def fuzzy_dark_membership(roi):
    values = roi.astype(np.float32).reshape(-1, 1)
    percentiles = np.linspace(15, 85, FCM_CLUSTERS)
    centers = np.percentile(values, percentiles).astype(np.float32).reshape(-1)

    for _ in range(FCM_MAX_ITERATIONS):
        distances = np.abs(values - centers[None, :]) + 1e-6
        inverse = distances ** (-2.0 / (FCM_FUZZINESS - 1.0))
        memberships = inverse / np.sum(inverse, axis=1, keepdims=True)
        weights = memberships ** FCM_FUZZINESS
        updated = np.sum(weights * values, axis=0) / (
            np.sum(weights, axis=0) + 1e-6
        )
        if np.max(np.abs(updated - centers)) < 0.01:
            centers = updated
            break
        centers = updated

    darkest_cluster = int(np.argmin(centers))
    return memberships[:, darkest_cluster].reshape(roi.shape)


def vertical_diameter(mask):
    widths = []
    for row in range(mask.shape[0]):
        xs = np.flatnonzero(mask[row] > 0)
        if xs.size >= 3:
            widths.append(int(xs[-1] - xs[0] + 1))
    return float(np.median(widths)) if len(widths) >= 5 else float("nan")


def overlay(image, mask):
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    selected = mask > 0
    result[selected] = (
        0.55 * result[selected] + 0.45 * np.array([0, 255, 255])
    ).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def main():
    region_image = read_frame(REGION_INPUT_DIR)
    fcm_image = read_frame(FCM_INPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x0, x1, y0, y1 = ROI_XYXY
    center_global = VESSEL_CENTER_XY
    center_local = (center_global[0] - x0, center_global[1] - y0)
    prior = circular_prior(
        (y1-y0, x1-x0), center_local, PRIOR_RADIUS
    )

    # Region Growing input: Butterworth D0=0.50/order=10 + CLAHE.
    region_roi = region_image[y0:y1, x0:x1]
    core_roi = region_growing_core(region_roi, center_local)

    # FCM input: Gaussian K=5/sigma=0.1 + global histogram equalization.
    fcm_roi = fcm_image[y0:y1, x0:x1]
    membership = fuzzy_dark_membership(fcm_roi)
    fuzzy_candidate = (
        membership >= FCM_EDGE_MEMBERSHIP_THRESHOLD
    ).astype(np.uint8) * 255
    fuzzy_candidate[~prior] = 0
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fuzzy_candidate = cv2.morphologyEx(
        fuzzy_candidate, cv2.MORPH_CLOSE, close_kernel
    )
    fuzzy_candidate = choose_component(
        fuzzy_candidate, center_local, overlap_reference=core_roi
    )

    # Let FCM re-evaluate a narrow shell around the grown core boundary.
    # The shell extends approximately 6 px inward/outward from the core boundary.
    boundary_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (EDGE_REFINEMENT_KERNEL, EDGE_REFINEMENT_KERNEL),
    )
    inner_core = cv2.erode(core_roi, boundary_kernel, iterations=1)
    if not np.any(inner_core):
        inner_core = core_roi.copy()
    boundary_outer = cv2.dilate(core_roi, boundary_kernel, iterations=1) > 0
    boundary_inner = cv2.erode(core_roi, boundary_kernel, iterations=1) > 0
    refinement_band = boundary_outer & ~boundary_inner
    fuzzy_edge = np.zeros_like(fuzzy_candidate)
    fuzzy_edge[(fuzzy_candidate > 0) & refinement_band] = 255
    hybrid_roi = cv2.bitwise_or(inner_core, fuzzy_edge)
    hybrid_roi = cv2.morphologyEx(
        hybrid_roi, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    hybrid_roi[~prior] = 0

    def to_full(mask_roi):
        full = np.zeros((200, 200), dtype=np.uint8)
        full[y0:y1, x0:x1] = mask_roi
        return full

    core_mask = to_full(core_roi)
    fcm_candidate_mask = to_full(fuzzy_candidate)
    edge_mask = to_full(fuzzy_edge)
    hybrid_mask = to_full(hybrid_roi)
    hybrid_overlay_fcm = overlay(fcm_image, hybrid_mask)

    # Save intermediate and final masks to make the fusion auditable.
    named_masks = {
        "region_growing_core": core_mask,
        "fcm_fuzzy_candidate": fcm_candidate_mask,
        "fcm_edge_band": edge_mask,
        "hybrid_mask": hybrid_mask,
    }
    for name, mask in named_masks.items():
        cv2.imwrite(str(OUTPUT_DIR / f"{name}.png"), mask)
    cv2.imwrite(
        str(OUTPUT_DIR / "hybrid_overlay_on_fcm_input.png"),
        cv2.cvtColor(hybrid_overlay_fcm, cv2.COLOR_RGB2BGR),
    )

    panels = [
        ("Region Growing input\nButterworth + CLAHE", region_image),
        ("Region Growing core", core_mask),
        ("Region Growing core\n(overlay)", overlay(region_image, core_mask)),

        ("FCM input\nGaussian + Histogram Equalization", fcm_image),
        ("FCM fuzzy candidate", fcm_candidate_mask),
        ("FCM edge band", edge_mask),

        ("Hybrid mask", hybrid_mask),
        ("Hybrid overlay\non Region Growing input", overlay(region_image, hybrid_mask)),
        ("Hybrid overlay\non FCM input", hybrid_overlay_fcm),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11, 13))
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(
            image,
            cmap="gray" if image.ndim == 2 else None,
            vmin=0 if image.ndim == 2 else None,
            vmax=255 if image.ndim == 2 else None,
        )
        ax.set_title(title,fontsize=12)
        ax.axis("off")
    
    fig.suptitle(
        f"Hybrid Region Growing + FCM Segmentation (frame {FRAME_INDEX})\n"
        "Region Growing Core from Butterworth + CLAHE; FCM Edge from Gaussian + Histogram Equalization",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    figure_path = OUTPUT_DIR / f"hybrid_segmentation_frame{FRAME_INDEX:04d}.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    with (OUTPUT_DIR / "hybrid_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "component", "area_px", "vertical_diameter_px",
            "vertical_diameter_mm",
        ])
        for name, mask in named_masks.items():
            diameter_px = vertical_diameter(mask)
            writer.writerow([
                name,
                int(np.count_nonzero(mask)),
                diameter_px,
                diameter_px * MM_PER_PIXEL if np.isfinite(diameter_px) else "",
            ])

    print(f"Hybrid result figure saved to: {figure_path}")
    print("Inspect the core, FCM edge band, and hybrid overlay before using it for the cine.")


if __name__ == "__main__":
    main()
