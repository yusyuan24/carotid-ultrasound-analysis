"""Region Growing core + band-limited FCM + top-notch-repaired lumen mask.

The script processes a representative frame, sweeps Region Growing tolerance,
and checks core boundary failures across the available cine frames. It is not
run automatically.
"""
from pathlib import Path
from collections import deque
import csv
import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(os.path.normpath(str(Path(__file__).resolve().parent)))
FRAME_INDEX = 0
REGION_INPUT_DIR = (
    ROOT / "contrast_results" / "butterworth_D0.50_order10"
    / "adapthisteq_clahe"
)
FCM_INPUT_DIR = (
    ROOT / "contrast_results" / "gaussian_k5_sigma1.0"
    / "histogram_equalization"
)
OUTPUT_DIR = ROOT / "segmentation_rg_fcm_repair_results"

# Seed and ROI coordinates are relative to the 200 x 200 cropped frame.
VESSEL_CENTER_XY = (106, 103)  # dark-center fallback seed
UPPER_GRAY_SEED_XY = (103, 90)
ROI_XYXY = (31, 181, 28, 178)  # x0, x1, y0, y1
MAX_GROW_RADIUS_PX = 65
RG_SMOOTH_SIGMA = 2.0
RG_SEED_PATCH_HALF = 5

# Used only if neither seed produces a plateau in the target area range.
REGION_GROW_TOLERANCE = 35
TOLERANCE_SWEEP = tuple(range(20, 91, 5))
PLATEAU_AREA_RANGE_PX = (8000, 10000)
MIN_CORE_AREA_PX = 5000
MAX_CORE_AREA_PX = 13000
SEED_SCENARIOS = {
    "dark_center": VESSEL_CENTER_XY,
    "upper_gray": UPPER_GRAY_SEED_XY,
}

# RG morphology: opening separates narrow bridges before component selection.
RG_OPEN_KERNEL = 13
RG_CLOSE_KERNEL = 5

# FCM settings and boundary-band width.
FCM_CLUSTERS = 4
FCM_FUZZINESS = 2.0
FCM_MAX_ITERATIONS = 100
FCM_EDGE_MEMBERSHIP_THRESHOLD = 0.25
FCM_EDGE_BAND_RADIUS_PX = 8
FCM_CLOSE_KERNEL = 3

# Binary cleanup kernels after merging RG and FCM.
MERGE_OPEN_KERNEL = 9
MERGE_CLOSE_KERNEL = 5
FCM_HULL_EXPANSION_PX = 3
# Maximum distance that FCM cleanup may expand beyond the cleaned RG mask.
FCM_MAX_EXPANSION_PX = 2
MM_PER_PIXEL = 0.062397
# Repair shallow concavities only near the vessel's upper edge. Keeping the
# hull out of the lower boundary avoids turning the whole lumen into an ellipse.
TOP_NOTCH_REPAIR_FRACTION = 0.2


def save_figure(fig, path, **kwargs):
    """Remove a generated PNG before replacing it, avoiding stale-file errors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    with path.open("wb") as output_file:
        fig.savefig(output_file, format="png", **kwargs)


def frame_path(folder, index):
    return folder / f"frame_{index:04d}.png"


def read_frame(folder, index):
    path = frame_path(folder, index)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    if image.shape != (200, 200):
        raise ValueError(f"Expected 200x200 input, got {image.shape} at {path}")
    return image


def select_seed_component(mask, seed_xy):
    """Prefer the seed-containing component; fall back to the nearest component."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    seed_x, seed_y = map(int, seed_xy)
    if not (0 <= seed_x < binary.shape[1] and 0 <= seed_y < binary.shape[0]):
        return np.zeros_like(binary)
    seed_label = int(labels[seed_y, seed_x])
    if 0 < seed_label < count:
        return (labels == seed_label).astype(np.uint8) * 255

    seed_component_candidates = []
    for label in range(1, count):
        cx, cy = centroids[label]
        distance = float(np.hypot(cx - seed_x, cy - seed_y))
        area = int(stats[label, cv2.CC_STAT_AREA])
        seed_component_candidates.append((distance, -area, label))
    if not seed_component_candidates:
        return np.zeros_like(binary)
    nearest_label = min(seed_component_candidates)[2]
    return (labels == nearest_label).astype(np.uint8) * 255


def region_growing_core(roi, seed_xy, tolerance):
    """Smooth image, use a patch median and one-sided threshold for seeded RG."""
    seed_x, seed_y = map(int, seed_xy)
    height, width = roi.shape
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ValueError("Seed point falls outside the selected ROI.")

    smooth = cv2.GaussianBlur(roi, (0, 0), RG_SMOOTH_SIGMA)
    h = RG_SEED_PATCH_HALF
    patch = smooth[max(seed_y - h, 0):seed_y + h + 1,
                   max(seed_x - h, 0):seed_x + h + 1]
    limit = int(np.median(patch)) + tolerance
    visited = np.zeros((height, width), dtype=bool)
    grown = np.zeros((height, width), dtype=np.uint8)
    queue = deque([(seed_x, seed_y)])
    visited[seed_y, seed_x] = True
    max_radius_sq = MAX_GROW_RADIUS_PX ** 2

    while queue:
        x, y = queue.popleft()
        # This is a generous growth limit, not a fixed-size output mask.
        if (x - seed_x) ** 2 + (y - seed_y) ** 2 > max_radius_sq:
            continue
        if int(smooth[y, x]) > limit:
            continue
        grown[y, x] = 255
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (RG_OPEN_KERNEL, RG_OPEN_KERNEL)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (RG_CLOSE_KERNEL, RG_CLOSE_KERNEL)
    )
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, close_kernel)
    grown = fill_holes(grown)
    grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, open_kernel)
    return select_seed_component(grown, (seed_x, seed_y))


def touches_roi_border(mask):
    return bool(
        np.any(mask[0, :]) or np.any(mask[-1, :])
        or np.any(mask[:, 0]) or np.any(mask[:, -1])
    )


def darkest_fcm_membership(image):
    """Return each pixel's membership in the darkest FCM intensity cluster."""
    values = image.astype(np.float32).reshape(-1, 1)
    initial_percentiles = np.linspace(15, 85, FCM_CLUSTERS)
    centers = np.percentile(values, initial_percentiles).astype(np.float32).reshape(-1)
    memberships = None
    for _ in range(FCM_MAX_ITERATIONS):
        distances = np.abs(values - centers[None, :]) + 1e-6
        inverse_distances = distances ** (-2.0 / (FCM_FUZZINESS - 1.0))
        memberships = inverse_distances / np.sum(
            inverse_distances, axis=1, keepdims=True
        )
        weights = memberships ** FCM_FUZZINESS
        updated = np.sum(weights * values, axis=0) / (
            np.sum(weights, axis=0) + 1e-6
        )
        if np.max(np.abs(updated - centers)) < 0.01:
            centers = updated
            break
        centers = updated
    darkest_cluster = int(np.argmin(centers))
    return memberships[:, darkest_cluster].reshape(image.shape)


def fill_holes(mask):
    binary = (mask > 0).astype(np.uint8)
    flood_filled = binary.copy()
    flood_mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), np.uint8)
    cv2.floodFill(flood_filled, flood_mask, (0, 0), 1)
    holes = flood_filled == 0
    return ((binary > 0) | holes).astype(np.uint8) * 255


def keep_largest_component(mask):
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8) * 255


def convex_hull_mask(mask):
    """Fill the convex hull of the largest foreground contour."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    hull_mask = np.zeros(mask.shape, dtype=np.uint8)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        cv2.fillConvexPoly(hull_mask, hull, 255)
    return hull_mask


def repair_top_notch(mask, max_depth_fraction=TOP_NOTCH_REPAIR_FRACTION):
    """Fill only the shallow top concavity, preserving the measured side/bottom contour."""
    binary = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return binary
    top = int(ys.min())
    bottom = int(ys.max())
    depth = max(1, int(round((bottom - top + 1) * max_depth_fraction)))
    hull = convex_hull_mask(binary) > 0
    repaired = binary > 0
    # The hull closes the crown notch, but is limited to a shallow top band;
    # all pixels below that band remain exactly as segmented.
    band_bottom = min(binary.shape[0], top + depth + 1)
    repaired[top:band_bottom] |= hull[top:band_bottom]
    return repaired.astype(np.uint8) * 255


def clean_merged_mask(mask, open_kernel, close_kernel):
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    cleaned = fill_holes(cleaned)
    return keep_largest_component(cleaned)


def fit_ellipse_trimmed(mask, iters=3, k=2.0, min_tol=0.04):
    """Iteratively fit an ellipse while trimming contour points with large residuals."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("No contour available for ellipse fitting.")
    points = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(points) < 20:
        raise ValueError("Ellipse fitting requires at least 20 contour points.")

    keep = np.ones(len(points), dtype=bool)
    for _ in range(iters):
        if np.count_nonzero(keep) < 5:
            raise ValueError("Too few inlier points remain for ellipse fitting.")
        ellipse = cv2.fitEllipse(points[keep].reshape(-1, 1, 2))
        (cx, cy), (axis_a, axis_b), angle_degrees = ellipse
        theta = np.deg2rad(angle_degrees)
        delta = points - np.array([cx, cy], dtype=np.float32)
        xr = delta[:, 0] * np.cos(theta) + delta[:, 1] * np.sin(theta)
        yr = -delta[:, 0] * np.sin(theta) + delta[:, 1] * np.cos(theta)
        residual = np.abs(
            np.hypot(xr / (axis_a / 2.0), yr / (axis_b / 2.0)) - 1.0
        )
        new_keep = residual <= max(min_tol, k * float(np.median(residual)))
        if np.count_nonzero(new_keep) < 20 or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    if np.count_nonzero(keep) < 5:
        raise ValueError("Too few inlier points remain for final ellipse fit.")
    ellipse = cv2.fitEllipse(points[keep].reshape(-1, 1, 2))
    return ellipse, float(np.mean(keep))


def rasterize_ellipse(shape, ellipse):
    center_xy, axes_xy, angle_degrees = ellipse
    fitted = np.zeros(shape, dtype=np.uint8)
    center_int = tuple(np.round(center_xy).astype(int))
    radii_int = tuple(np.maximum(1, np.round(np.asarray(axes_xy) / 2).astype(int)))
    cv2.ellipse(fitted, center_int, radii_int, angle_degrees, 0, 360, 255, -1)
    return fitted


def mask_iou(mask_a, mask_b):
    a = mask_a > 0
    b = mask_b > 0
    union = np.count_nonzero(a | b)
    if union == 0:
        return float("nan")
    return float(np.count_nonzero(a & b) / union)


def to_full_frame(mask_roi):
    x0, x1, y0, y1 = ROI_XYXY
    full = np.zeros((200, 200), dtype=np.uint8)
    full[y0:y1, x0:x1] = mask_roi
    return full


def overlay(image, mask):
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    selected = mask > 0
    result[selected] = (
        0.55 * result[selected] + 0.45 * np.array([0, 255, 255])
    ).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def save_overlay(image, mask, output_path, title):
    """Save one mask overlay with a visible contour outline."""
    rendered = overlay(image, mask)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rendered)
    if np.any(mask):
        ax.contour(mask > 0, levels=[0.5], colors=["cyan"], linewidths=1.2)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=300,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def segment_frame(region_image, fcm_image, seed_local, tolerance):
    """Run RG, band-limited FCM, cleanup, trimmed ellipse fit, and QC for one frame."""
    x0, x1, y0, y1 = ROI_XYXY
    region_roi = region_image[y0:y1, x0:x1]
    fcm_roi = fcm_image[y0:y1, x0:x1]

    core = region_growing_core(region_roi, seed_local, tolerance)
    core_area = int(np.count_nonzero(core))
    border = touches_roi_border(core)

    membership = darkest_fcm_membership(fcm_roi)
    candidate = (membership >= FCM_EDGE_MEMBERSHIP_THRESHOLD).astype(np.uint8) * 255
    fcm_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (FCM_CLOSE_KERNEL, FCM_CLOSE_KERNEL)
    )
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, fcm_close)

    band_size = 2 * FCM_EDGE_BAND_RADIUS_PX + 1
    band_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_size, band_size)
    )
    outer = cv2.dilate(core, band_kernel, iterations=1) > 0
    inner = cv2.erode(core, band_kernel, iterations=1) > 0
    band = outer & ~inner
    hull = convex_hull_mask(core)
    hull_expansion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * FCM_HULL_EXPANSION_PX + 1, 2 * FCM_HULL_EXPANSION_PX + 1),
    )
    allowed = cv2.dilate(hull, hull_expansion_kernel, iterations=1) > 0
    fcm_edge = ((candidate > 0) & band & allowed).astype(np.uint8) * 255

    merged = cv2.bitwise_or(core, fcm_edge)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MERGE_OPEN_KERNEL, MERGE_OPEN_KERNEL)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MERGE_CLOSE_KERNEL, MERGE_CLOSE_KERNEL)
    )
    cleaned_without_fcm = clean_merged_mask(core, open_kernel, close_kernel)
    cleaned = clean_merged_mask(merged, open_kernel, close_kernel)
    # FCM may refine the boundary, but cleanup must not create a remote bulge.
    # Keep its contribution within a narrow dilation of the RG-only result.
    fcm_limit_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * FCM_MAX_EXPANSION_PX + 1, 2 * FCM_MAX_EXPANSION_PX + 1),
    )
    fcm_allowed = cv2.dilate(cleaned_without_fcm, fcm_limit_kernel, iterations=1)
    cleaned = cv2.bitwise_and(cleaned, fcm_allowed)
    cleaned = cv2.bitwise_or(cleaned, cleaned_without_fcm)
    fcm_added_px = (
        int(np.count_nonzero(cleaned))
        - int(np.count_nonzero(cleaned_without_fcm))
    )
    fcm_added_fraction_of_core = (
        float(fcm_added_px / core_area) if core_area > 0 else float("nan")
    )

    fit_error = ""
    inlier_ratio = float("nan")
    iou = float("nan")
    ellipse = None
    try:
        ellipse, inlier_ratio = fit_ellipse_trimmed(cleaned)
        # Use the ellipse fit for quality checks only. Preserve the segmented
        # contour and repair its shallow top notch instead of rasterizing an
        # ellipse, which would erase the natural contour.
        final_roi = repair_top_notch(cleaned)
        iou = mask_iou(cleaned, final_roi)
    except (ValueError, cv2.error) as exc:
        final_roi = np.zeros_like(cleaned)
        fit_error = str(exc)

    reasons = []
    if core_area < MIN_CORE_AREA_PX:
        reasons.append("CORE_AREA_BELOW_MIN")
    if core_area > MAX_CORE_AREA_PX:
        reasons.append("CORE_AREA_ABOVE_MAX")
    if border:
        reasons.append("CORE_TOUCHES_ROI_BORDER")
    if ellipse is None:
        reasons.append("ELLIPSE_FIT_FAILED")
    elif iou < 0.85:
        reasons.append("ELLIPSE_IOU_BELOW_0.85")

    axes = (float("nan"), float("nan"))
    center = (float("nan"), float("nan"))
    angle = float("nan")
    if ellipse is not None:
        center, axes, angle = ellipse
        axes = tuple(sorted(map(float, axes), reverse=True))

    area = int(np.count_nonzero(final_roi))
    eq_diameter_px = (
        float(np.sqrt(4.0 * area / np.pi)) if area else float("nan")
    )
    result = {
        "core": core,
        "candidate": candidate,
        "band": band.astype(np.uint8) * 255,
        "fcm_edge": fcm_edge,
        "merged": merged,
        "cleaned_without_fcm": cleaned_without_fcm,
        "cleaned": cleaned,
        "cleaned_without_fcm_area_px": int(np.count_nonzero(cleaned_without_fcm)),
        "fcm_added_px": fcm_added_px,
        "fcm_added_fraction_of_core": fcm_added_fraction_of_core,
        "final": final_roi,
        "core_area_px": core_area,
        "core_touches_border": border,
        "inlier_ratio": inlier_ratio,
        "iou": iou,
        "ellipse": ellipse,
        "ellipse_major_axis_px": axes[0],
        "ellipse_minor_axis_px": axes[1],
        "ellipse_angle_degrees": float(angle),
        "ellipse_center_x_roi_px": float(center[0]),
        "ellipse_center_y_roi_px": float(center[1]),
        "ellipse_area_px": area,
        "equivalent_diameter_px": eq_diameter_px,
        "equivalent_diameter_mm": eq_diameter_px * MM_PER_PIXEL,
        "quality_flag": "REVIEW" if reasons else "PASS",
        "quality_reason": ";".join(reasons),
        "diameter_usable": not reasons,
        "ellipse_fit_error": fit_error,
    }
    return result


def choose_plateau_settings(sweep_rows):
    """Choose the midpoint of the longest in-range tolerance plateau (>=2 points)."""
    lower_area, upper_area = PLATEAU_AREA_RANGE_PX
    plateau_runs = []
    for scenario, seed_xy in SEED_SCENARIOS.items():
        scenario_rows = [row for row in sweep_rows if row[0] == scenario]
        current_run = []
        runs = []
        for row in scenario_rows:
            _, tolerance, area, touches_border, empty = row
            eligible = (
                lower_area <= area <= upper_area
                and not touches_border
                and not empty
            )
            if eligible:
                current_run.append(tolerance)
            else:
                if current_run:
                    runs.append(current_run)
                current_run = []
        if current_run:
            runs.append(current_run)
        for run in runs:
            if len(run) >= 2:
                plateau_runs.append((len(run), scenario, seed_xy, run))

    if not plateau_runs:
        return "dark_center", VESSEL_CENTER_XY, REGION_GROW_TOLERANCE, False, []

    # Prefer the longest plateau; insertion order breaks ties in favor of dark_center.
    _, scenario, seed_xy, run = max(plateau_runs, key=lambda item: item[0])
    selected_tolerance = run[len(run) // 2]
    return scenario, seed_xy, selected_tolerance, True, run


def save_tolerance_sweep(region_image, output_dir):
    x0, x1, y0, y1 = ROI_XYXY
    roi = region_image[y0:y1, x0:x1]
    rows = []
    for scenario, seed_global in SEED_SCENARIOS.items():
        seed_local = (seed_global[0] - x0, seed_global[1] - y0)
        for tolerance in TOLERANCE_SWEEP:
            core = region_growing_core(roi, seed_local, tolerance)
            area = int(np.count_nonzero(core))
            border = touches_roi_border(core)
            empty = area == 0
            rows.append((scenario, tolerance, area, border, empty))

    selected = choose_plateau_settings(rows)

    with (output_dir / "tolerance_area_sweep.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "seed_scenario", "seed_x", "seed_y", "tolerance", "core_area_px",
            "touches_roi_border", "empty_core",
        ])
        for scenario, tolerance, area, border, empty in rows:
            seed_global = SEED_SCENARIOS[scenario]
            writer.writerow([
                scenario, seed_global[0], seed_global[1], tolerance,
                area, border, empty,
            ])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"dark_center": "#276FBF", "upper_gray": "#E07A27"}
    for scenario in SEED_SCENARIOS:
        scenario_rows = [row for row in rows if row[0] == scenario]
        ax.plot(
            [row[1] for row in scenario_rows],
            [row[2] for row in scenario_rows],
            marker="o", color=colors[scenario], label=f"{scenario} seed",
        )
    ax.axhspan(
        *PLATEAU_AREA_RANGE_PX, color="#70AD47", alpha=0.12,
        label="Target plateau area (8,000–10,000 px)",
    )
    scenario, seed_global, tolerance, plateau_found, plateau_run = selected
    ax.axvline(tolerance, color="#D1495B", ls="--",
               label=f"Selected tolerance: {tolerance} ({scenario})")
    ax.set(
        xlabel="Region Growing intensity tolerance",
        ylabel="Core area (pixels)",
        title=f"Frame {FRAME_INDEX}: core area versus tolerance and seed",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "tolerance_area_sweep.png", dpi=200)
    plt.close(fig)
    selection = {
        "seed_scenario": scenario,
        "seed_xy": list(seed_global),
        "tolerance": tolerance,
        "plateau_found": plateau_found,
        "plateau_tolerances": plateau_run,
    }
    return rows, selection


def save_frame0_outputs(region_image, fcm_image, result, output_dir):
    """Save detailed intermediate masks, overlays, and comparison for Frame 0."""
    full_masks = {
        "region_growing_core": to_full_frame(result["core"]),
        "fcm_candidate": to_full_frame(result["candidate"]),
        "fcm_refinement_band": to_full_frame(result["band"]),
        "fcm_edge_in_band": to_full_frame(result["fcm_edge"]),
        "merged_mask": to_full_frame(result["merged"]),
        "cleaned_without_fcm_mask": to_full_frame(result["cleaned_without_fcm"]),
        "cleaned_mask": to_full_frame(result["cleaned"]),
        "final_ellipse_mask": to_full_frame(result["final"]),
    }
    for name, mask in full_masks.items():
        cv2.imwrite(str(output_dir / f"{name}.png"), mask)

    # Save each processing stage as a standalone overlay, using the image that
    # supplied that stage's intensity information as its background.
    step_stages = [
        ("region_growing_core", "Region Growing core", region_image),
        ("fcm_candidate", "FCM candidate", fcm_image),
        ("fcm_refinement_band", "FCM refinement band", fcm_image),
        ("fcm_edge_in_band", "FCM edge within RG band", fcm_image),
        ("merged_mask", "RG + FCM merged mask", fcm_image),
        ("cleaned_without_fcm_mask", "RG-only cleaned mask (ablation)", region_image),
        ("cleaned_mask", "Cleaned RG + FCM mask", fcm_image),
        ("final_ellipse_mask", "Final Notch Repair mask", fcm_image),
    ]
    for key, title, background in step_stages:
        save_overlay(
            background,
            full_masks[key],
            output_dir / f"{key}_overlay.png",
            f"{title} overlay",
        )

    fig, axes = plt.subplots(
        len(step_stages), 2,
        figsize=(9, 3.2 * len(step_stages)),
        squeeze=False,
    )
    for row_index, (key, title, background) in enumerate(step_stages):
        mask = full_masks[key]
        mask_ax, overlay_ax = axes[row_index]
        mask_ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
        mask_ax.set_title(f"{title}: mask")
        mask_ax.axis("off")
        overlay_ax.imshow(overlay(background, mask))
        if np.any(mask):
            overlay_ax.contour(mask > 0, levels=[0.5], colors=["cyan"], linewidths=1.0)
        overlay_ax.set_title(f"{title}: overlay")
        overlay_ax.axis("off")
    fig.suptitle(f"Stepwise masks and overlays (frame {FRAME_INDEX})", y=1.0)
    fig.tight_layout()
    save_figure(fig, output_dir / "segmentation_stepwise_masks_and_overlays.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    merged_overlay = overlay(fcm_image, full_masks["merged_mask"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(merged_overlay)
    if np.any(full_masks["merged_mask"]):
        ax.contour(full_masks["merged_mask"] > 0, levels=[0.5],
                   colors=["cyan"], linewidths=1.2)
    ax.set_title("Merged RG + FCM mask (before cleanup and notch repair)")
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, output_dir / "merged_mask_overlay.png", dpi=300,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    final_overlay = overlay(fcm_image, full_masks["final_ellipse_mask"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(final_overlay)
    if np.any(full_masks["final_ellipse_mask"]):
        ax.contour(full_masks["final_ellipse_mask"] > 0, levels=[0.5],
                   colors=["cyan"], linewidths=1.2)
    ax.set_title("Final RG + Band-limited FCM + Notch Repair")
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, output_dir / "final_ellipse_overlay.png", dpi=300,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    comparison_stages = [
        ("region_growing_core", "Region Growing Core\nInput: Butterworth + CLAHE", region_image),
        ("fcm_candidate", "FCM Candidate\nInput: Gaussian + HE ", fcm_image),
        ("merged_mask", "RG + FCM Merged Mask", fcm_image),
        ("cleaned_without_fcm_mask", "RG-only Cleaned Mask (ablation)", region_image),
        ("cleaned_mask", "Cleaned RG + FCM Mask", fcm_image),
        ("final_ellipse_mask", "Final Notch Repair Mask", fcm_image),
    ]

    # Top row: four compact panels. Bottom row: two panels spanning two
    # columns each and twice the row height, so their image edges align with
    # the full width of the two top panels above them.
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1])
    positions = [
        grid[0, 0], grid[0, 1], grid[0, 2], grid[0, 3],
        grid[1:3, 0:2], grid[1:3, 2:4],
    ]
    for position, (key, title, background) in zip(positions, comparison_stages):
        ax = fig.add_subplot(position)
        ax.imshow(overlay(background, full_masks[key]))
        if np.any(full_masks[key]):
            ax.contour(full_masks[key] > 0, levels=[0.5],
                       colors=["cyan"], linewidths=1.0)
        ax.set_title(title, fontsize=16)
        ax.axis("off")
    fig.suptitle(f"RG + Band-limited FCM + Notch Repair (frame {FRAME_INDEX})", 
                 fontsize=24, y=0.99)
    fig.tight_layout()
    save_figure(fig, output_dir / "segmentation_rg_fcm_ellipse_comparison.png",
                dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview_dir = OUTPUT_DIR / f"frame_{FRAME_INDEX:04d}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    final_mask_dir = OUTPUT_DIR / "all_frame_masks"
    final_mask_dir.mkdir(parents=True, exist_ok=True)

    region_paths = sorted(REGION_INPUT_DIR.glob("frame_*.png"))
    if not region_paths:
        raise FileNotFoundError(f"No Region Growing inputs found in {REGION_INPUT_DIR}")
    fcm_names = {path.name for path in FCM_INPUT_DIR.glob("frame_*.png")}
    missing_fcm = [path.name for path in region_paths if path.name not in fcm_names]
    if missing_fcm:
        raise FileNotFoundError(
            f"Missing matching FCM input frames, e.g. {missing_fcm[:5]}"
        )

    region_image_0 = read_frame(REGION_INPUT_DIR, FRAME_INDEX)
    tolerance_rows, selection = save_tolerance_sweep(region_image_0, OUTPUT_DIR)
    selected_seed_global = tuple(selection["seed_xy"])
    selected_tolerance = int(selection["tolerance"])
    x0, x1, y0, y1 = ROI_XYXY
    seed_local = (selected_seed_global[0] - x0,
                  selected_seed_global[1] - y0)

    metric_rows = []
    core_failure_rows = []
    frame0_result = None
    frame0_fcm_image = None

    for region_path in region_paths:
        frame_index = int(region_path.stem.split("_")[-1])
        region_image = read_frame(REGION_INPUT_DIR, frame_index)
        fcm_image = read_frame(FCM_INPUT_DIR, frame_index)
        result = segment_frame(
            region_image, fcm_image, seed_local, selected_tolerance
        )
        final_full = to_full_frame(result["final"])
        cleaned_full = to_full_frame(result["cleaned"])
        cv2.imwrite(
            str(final_mask_dir / f"final_ellipse_mask_{frame_index:04d}.png"),
            final_full,
        )
        cv2.imwrite(
            str(final_mask_dir / f"cleaned_mask_{frame_index:04d}.png"),
            cleaned_full,
        )

        core_area = result["core_area_px"]
        border = result["core_touches_border"]
        empty = core_area == 0
        too_small = core_area < MIN_CORE_AREA_PX
        too_large = core_area > MAX_CORE_AREA_PX
        core_failure = too_small or too_large or border
        metric_rows.append({
            "frame": frame_index,
            "rg_core_area_px": core_area,
            "rg_touches_roi_border": border,
            "rg_empty": empty,
            "rg_below_min_area": too_small,
            "rg_above_max_area": too_large,
            "rg_core_failure": core_failure,
            "merged_area_px": int(np.count_nonzero(result["merged"])),
            "cleaned_without_fcm_area_px": result["cleaned_without_fcm_area_px"],
            "cleaned_area_px": int(np.count_nonzero(result["cleaned"])),
            "fcm_added_px": result["fcm_added_px"],
            "fcm_added_fraction_of_core": result["fcm_added_fraction_of_core"],
            "ellipse_area_px": result["ellipse_area_px"],
            "ellipse_inlier_ratio": result["inlier_ratio"],
            "cleaned_ellipse_iou": result["iou"],
            "quality_flag": result["quality_flag"],
            "quality_reason": result["quality_reason"],
            "diameter_usable": result["diameter_usable"],
            "equivalent_diameter_px": result["equivalent_diameter_px"],
            "equivalent_diameter_mm": result["equivalent_diameter_mm"],
            "ellipse_major_axis_px": result["ellipse_major_axis_px"],
            "ellipse_minor_axis_px": result["ellipse_minor_axis_px"],
            "ellipse_angle_degrees": result["ellipse_angle_degrees"],
            "ellipse_center_x_roi_px": result["ellipse_center_x_roi_px"],
            "ellipse_center_y_roi_px": result["ellipse_center_y_roi_px"],
            "ellipse_fit_error": result["ellipse_fit_error"],
        })
        core_failure_rows.append({
            "frame": frame_index,
            "core_area_px": core_area,
            "touches_roi_border": border,
            "empty_core": empty,
            "below_min_area": too_small,
            "above_max_area": too_large,
            "failure": core_failure,
        })

        if frame_index == FRAME_INDEX:
            frame0_result = result
            frame0_fcm_image = fcm_image

    if frame0_result is None:
        raise FileNotFoundError(f"Frame {FRAME_INDEX} is missing from the input cine.")

    metric_fields = list(metric_rows[0].keys())
    with (OUTPUT_DIR / "segmentation_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    failure_fields = list(core_failure_rows[0].keys())
    with (OUTPUT_DIR / "rg_core_failure_check.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows(core_failure_rows)

    save_frame0_outputs(
        region_image_0, frame0_fcm_image, frame0_result, preview_dir
    )

    low_iou_count = sum(
        np.isfinite(row["cleaned_ellipse_iou"])
        and row["cleaned_ellipse_iou"] < 0.85
        for row in metric_rows
    )
    fit_failure_count = sum(not np.isfinite(row["ellipse_inlier_ratio"])
                            for row in metric_rows)
    usable_count = sum(bool(row["diameter_usable"]) for row in metric_rows)
    total_fcm_added_px = sum(row["fcm_added_px"] for row in metric_rows)
    total_rg_core_area_px = sum(row["rg_core_area_px"] for row in metric_rows)
    frame_fcm_added_fractions = [
        row["fcm_added_fraction_of_core"] for row in metric_rows
        if np.isfinite(row["fcm_added_fraction_of_core"])
    ]
    mean_fcm_added_fraction = (
        float(np.mean(frame_fcm_added_fractions))
        if frame_fcm_added_fractions else None
    )
    summary = {
        "frames_processed": len(metric_rows),
        "selected_seed_scenario": selection["seed_scenario"],
        "selected_seed_xy": list(selected_seed_global),
        "selected_region_grow_tolerance": selected_tolerance,
        "tolerance_plateau_found": selection["plateau_found"],
        "plateau_tolerances": selection["plateau_tolerances"],
        "tolerance_sweep_values": list(TOLERANCE_SWEEP),
        "plateau_area_range_px": list(PLATEAU_AREA_RANGE_PX),
        "min_core_area_px": MIN_CORE_AREA_PX,
        "max_core_area_px": MAX_CORE_AREA_PX,
        "roi_border_touch_count": sum(row["touches_roi_border"] for row in core_failure_rows),
        "empty_core_count": sum(row["empty_core"] for row in core_failure_rows),
        "below_min_area_count": sum(row["below_min_area"] for row in core_failure_rows),
        "above_max_area_count": sum(row["above_max_area"] for row in core_failure_rows),
        "core_failure_count": sum(row["failure"] for row in core_failure_rows),
        "ellipse_fit_failure_count": int(fit_failure_count),
        "cleaned_ellipse_iou_below_0_85_count": int(low_iou_count),
        "diameter_usable_frame_count": int(usable_count),
        "fcm_hull_expansion_px": FCM_HULL_EXPANSION_PX,
        "fcm_added_px_total": int(total_fcm_added_px),
        "rg_core_area_px_total": int(total_rg_core_area_px),
        "fcm_added_fraction_of_total_rg_core_area": (
            float(total_fcm_added_px / total_rg_core_area_px)
            if total_rg_core_area_px else None
        ),
        "mean_frame_fcm_added_fraction_of_core": mean_fcm_added_fraction,
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Segmentation outputs saved in {OUTPUT_DIR}.")
    print("Review tolerance_area_sweep.png, segmentation_metrics.csv, and run_summary.json.")


if __name__ == "__main__":
    main()
