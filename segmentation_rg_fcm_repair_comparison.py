"""Compare the cleaned RG+FCM mask with the final notch-repaired mask.

The comparison processes the available aligned cine frames and includes
before/after mask-area diameter, boundary-gradient, and optional manual-label
measurements. It does not run automatically.
"""
from pathlib import Path
import csv
import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks, savgol_filter


ROOT = Path(os.path.normpath(str(Path(__file__).resolve().parent)))
REFERENCE_DIR = ROOT / "aligned_frames"
MASK_DIR = ROOT / "segmentation_rg_fcm_repair_results" / "all_frame_masks"
METRICS_CSV = ROOT / "segmentation_rg_fcm_repair_results" / "segmentation_metrics.csv"
MANUAL_MASK_DIR = ROOT / "manual_masks"  # Optional frame_XXXX.png binary labels.
OUTPUT_DIR = ROOT / "segmentation_rg_fcm_repair_comparison"

MM_PER_PIXEL = 0.062397
SYSTOLIC_PRESSURE_MMHG = 105.0
DIASTOLIC_PRESSURE_MMHG = 70.0
MIN_BEAT_PERIOD_S = 0.6
BEAT_PEAK_PROMINENCE_MM = 0.05
EDGE_FRAMES = 15
SG_WINDOW_LENGTH = 31
SG_POLYORDER = 3


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


def draw_diameter_curve(ax, times, diameters, signal, title, color,
                        show_legend=True):
    """Draw one method with peak/valley markers and its diameter extrema."""
    raw_color = "#f6c39f" if color == "#e8751a" else "#a9c9ee"
    # Include the method name in the legend as well as the subplot title.
    ax.plot([], [], color=color, lw=3.0, label=title)
    ax.plot(times, diameters, color=raw_color, lw=0.9, alpha=0.8,
            label="Measured diameter")
    ax.plot(times, signal["smoothed"], color=color, lw=2.0,
            label="Smoothed diameter (SG)")
    peaks = signal["peaks"]
    valleys = signal["valleys"]
    ax.scatter(times[peaks], signal["smoothed"][peaks], color="#d94b5b",
               s=90, zorder=4, label="Peaks")
    ax.scatter(times[valleys], signal["smoothed"][valleys], color="#2d8b57",
               s=90, zorder=4, label="Valleys")
    finite_smoothed = signal["smoothed"][np.isfinite(signal["smoothed"])]
    if finite_smoothed.size:
        maximum = float(np.max(finite_smoothed))
        # Anchor D_min to the lowest detected valley, even if an edge frame
        # outside the detected beats happens to be numerically lower.
        minimum = (
            float(np.min(signal["smoothed"][valleys]))
            if len(valleys)
            else float(np.min(finite_smoothed))
        )
        ax.axhline(maximum, color="#d94b5b", ls="--", lw=1.2,
                   label=rf"$D_{{max}}$ = {maximum:.3f} mm")
        ax.axhline(minimum, color="#2d8b57", ls="--", lw=1.2,
                   label=rf"$D_{{min}}$ = {minimum:.3f} mm")
        if not show_legend:
            label_box = {"facecolor": "white", "alpha": 0.85,
                         "edgecolor": "none", "pad": 1.5}
            ax.annotate(
                rf"$D_{{max}}$ = {maximum:.3f} mm",
                xy=(times[-1], maximum), xytext=(-5, 5),
                textcoords="offset points", ha="right", va="bottom",
                color="#d94b5b", fontsize=12, zorder=5,
                bbox=label_box,
            )
            ax.annotate(
                rf"$D_{{min}}$ = {minimum:.3f} mm",
                xy=(times[-1], minimum), xytext=(-5, -5),
                textcoords="offset points", ha="right", va="top",
                color="#2d8b57", fontsize=12, zorder=5,
                bbox=label_box,
            )
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_ylabel("Lumen diameter (mm)", fontsize=16)
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(fontsize=11, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.28), frameon=False)


def save_diameter_curve(times, diameters, signal, title, output_path, color):
    """Save one method's diameter curve in the carotid_analysis.py style."""
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_diameter_curve(ax, times, diameters, signal, title, color)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def boundary_edge_strength(gray, mask):
    """Mean Sobel magnitude on the largest mask boundary (proxy only)."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.5).astype(np.float32)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return float("nan")
    points = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return float(magnitude[points[:, 1], points[:, 0]].mean())


def equivalent_diameter_mm(mask):
    area = int(np.count_nonzero(mask))
    if area == 0:
        return float("nan")
    return float(np.sqrt(4.0 * area / np.pi) * MM_PER_PIXEL)


def dice_score(mask_a, mask_b):
    a, b = mask_a > 0, mask_b > 0
    denominator = np.count_nonzero(a) + np.count_nonzero(b)
    return float(2 * np.count_nonzero(a & b) / denominator) if denominator else float("nan")


def read_frame_time_ms():
    import pydicom
    dicom = ROOT / "CCA DICOM" / "00000000"
    dataset = pydicom.dcmread(dicom, stop_before_pixels=True)
    return float(dataset.FrameTime)


def detect_beats(smoothed, frame_ms, min_period_s=MIN_BEAT_PERIOD_S,
                 edge_frames=EDGE_FRAMES):
    """Detect peaks first, then one valley between each adjacent peak pair."""
    fs = 1000.0 / frame_ms
    peaks, _ = find_peaks(
        smoothed,
        distance=max(1, int(min_period_s * fs)),
        prominence=BEAT_PEAK_PROMINENCE_MM,
    )
    valleys = np.asarray([
        a + int(np.argmin(smoothed[a:b + 1]))
        for a, b in zip(peaks[:-1], peaks[1:])
    ], dtype=int)
    n = len(smoothed)
    peaks = peaks[(peaks >= edge_frames) & (peaks < n - edge_frames)]
    valleys = valleys[(valleys >= edge_frames) & (valleys < n - edge_frames)]
    return peaks, valleys


def summarize_signal(diameters, valid, frame_time_ms):
    indexes = np.arange(len(diameters))
    valid = valid & np.isfinite(diameters)
    if np.count_nonzero(valid) < 10:
        raise ValueError("Fewer than 10 valid frames for signal comparison.")
    filled = np.interp(indexes, indexes[valid], diameters[valid])
    window = min(SG_WINDOW_LENGTH, len(filled) if len(filled) % 2 else len(filled) - 1)
    if window <= SG_POLYORDER:
        raise ValueError("Too few frames for Savitzky-Golay smoothing.")
    smoothed_full = savgol_filter(filled, window_length=window, polyorder=SG_POLYORDER)
    smoothed = smoothed_full.copy()
    smoothed[~valid] = np.nan
    rms = float(np.sqrt(np.mean((diameters[valid] - smoothed_full[valid]) ** 2)))
    peaks, valleys = detect_beats(smoothed_full, frame_time_ms)
    peaks = peaks[valid[peaks]]
    valleys = valleys[valid[valleys]]
    mean_peak = float(np.mean(smoothed_full[peaks])) if len(peaks) else float("nan")
    mean_valley = float(np.mean(smoothed_full[valleys])) if len(valleys) else float("nan")
    amplitude = mean_peak - mean_valley if len(peaks) and len(valleys) else float("nan")
    relative_change = amplitude / mean_valley if np.isfinite(amplitude) and mean_valley > 0 else float("nan")
    log_diameter_ratio = (
        float(np.log(mean_peak / mean_valley))
        if len(peaks) and len(valleys) and mean_valley > 0 else float("nan")
    )
    return {
        "smoothed": smoothed,
        "rms_raw_minus_sg_mm": rms,
        "pulse_amplitude_mean_peak_minus_valley_mm": float(amplitude),
        "peak_count": int(len(peaks)),
        "valley_count": int(len(valleys)),
        "detected_cycle_count": int(min(len(peaks), len(valleys))),
        "mean_peak_interval_frames": (
            float(np.mean(np.diff(peaks))) if len(peaks) > 1 else float("nan")
        ),
        "estimated_pulse_rate_bpm": (
            float(60_000.0 / (np.mean(np.diff(peaks)) * frame_time_ms))
            if len(peaks) > 1 and np.mean(np.diff(peaks)) > 0 else float("nan")
        ),
        "mean_peak_diameter_mm": mean_peak,
        "mean_valley_diameter_mm": mean_valley,
        "relative_diameter_change": float(relative_change),
        "log_peak_over_valley_diameter": log_diameter_ratio,
        "log_systolic_over_diastolic_pressure": float(
            np.log(SYSTOLIC_PRESSURE_MMHG / DIASTOLIC_PRESSURE_MMHG)
        ),
        "peaks": peaks,
        "valleys": valleys,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(REFERENCE_DIR.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No aligned frames found in {REFERENCE_DIR}")
    if not METRICS_CSV.is_file():
        raise FileNotFoundError(f"Missing segmentation metrics: {METRICS_CSV}")
    frame_time_ms = read_frame_time_ms()
    with METRICS_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        metrics = {int(row["frame"]): row for row in csv.DictReader(file)}

    before_masks, after_masks = [], []
    before_diameters, after_diameters, after_valid = [], [], []
    before_edges, after_edges = [], []
    rows, manual_metrics = [], []

    for path in frame_paths:
        frame = int(path.stem.split("_")[-1])
        image = read_gray(path)
        before_path = MASK_DIR / f"cleaned_mask_{frame:04d}.png"
        after_path = MASK_DIR / f"final_ellipse_mask_{frame:04d}.png"
        before = read_gray(before_path)
        after = read_gray(after_path)
        if before.shape != image.shape or after.shape != image.shape:
            raise ValueError(f"Image/mask dimensions differ for frame {frame}.")
        before_masks.append(before)
        after_masks.append(after)

        before_d = equivalent_diameter_mm(before)
        after_d = equivalent_diameter_mm(after)
        metric = metrics.get(frame, {})
        after_ok = metric.get("diameter_usable", "True").strip().lower() in ("true", "1", "yes")
        after_valid.append(bool(after_ok and np.isfinite(after_d)))
        before_diameters.append(before_d)
        after_diameters.append(after_d)
        before_edge = boundary_edge_strength(image, before)
        after_edge = boundary_edge_strength(image, after)
        before_edges.append(before_edge)
        after_edges.append(after_edge)

        manual_path = MANUAL_MASK_DIR / f"frame_{frame:04d}.png"
        dice_before = dice_after = err_before = err_after = ""
        if manual_path.is_file():
            manual = read_gray(manual_path)
            manual_d = equivalent_diameter_mm(manual)
            dice_before = dice_score(before, manual)
            dice_after = dice_score(after, manual)
            err_before = abs(before_d - manual_d)
            err_after = abs(after_d - manual_d)
            manual_metrics.append({
                "frame": frame,
                "dice_before_fit": dice_before,
                "dice_after_fit": dice_after,
                "diameter_abs_error_before_fit_mm": err_before,
                "diameter_abs_error_after_fit_mm": err_after,
            })

        rows.append({
            "frame": frame,
            "time_s": frame * frame_time_ms / 1000.0,
            "cleaned_area_px_before_fit": int(np.count_nonzero(before)),
            "ellipse_area_px_after_fit": int(np.count_nonzero(after)),
            "equivalent_diameter_before_fit_mm": before_d,
            "equivalent_diameter_after_fit_mm": after_d,
            "boundary_gradient_before_fit": before_edge,
            "boundary_gradient_after_fit": after_edge,
            "ellipse_inlier_ratio": metric.get("ellipse_inlier_ratio", ""),
            "cleaned_ellipse_iou": metric.get("cleaned_ellipse_iou", ""),
            "quality_flag": metric.get("quality_flag", ""),
            "diameter_usable_after_fit": after_ok,
            "manual_dice_before_fit": dice_before,
            "manual_dice_after_fit": dice_after,
            "manual_diameter_abs_error_before_fit_mm": err_before,
            "manual_diameter_abs_error_after_fit_mm": err_after,
        })

    before_diameters = np.asarray(before_diameters, dtype=float)
    after_diameters = np.asarray(after_diameters, dtype=float)
    before_signal = summarize_signal(
        before_diameters, np.isfinite(before_diameters), frame_time_ms
    )
    after_signal = summarize_signal(
        after_diameters, np.asarray(after_valid, dtype=bool), frame_time_ms
    )
    for index, row in enumerate(rows):
        row["diameter_smoothed_before_fit_mm"] = float(before_signal["smoothed"][index])
        row["diameter_smoothed_after_fit_mm"] = float(after_signal["smoothed"][index])

    times = np.asarray([row["time_s"] for row in rows])
    with (OUTPUT_DIR / "ellipse_fit_comparison_timeseries.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    before_edge_mean = float(np.nanmean(before_edges))
    after_edge_mean = float(np.nanmean(after_edges))
    summary = {
        "frames_compared": len(rows),
        "comparison": "cleaned RG+FCM mask before notch repair vs final notch-repaired mask",
        "reference_for_boundary_gradient": "aligned_frames (same image for both masks)",
        "boundary_gradient_is_proxy_only": True,
        "diameter_definition": "Equivalent diameter sqrt(4A/pi) applied to both masks.",
        "frame_time_ms": frame_time_ms,
        "beat_detection": {
            "method": "peaks first; one minimum valley between adjacent peaks",
            "minimum_period_s": MIN_BEAT_PERIOD_S,
            "peak_prominence_mm": BEAT_PEAK_PROMINENCE_MM,
            "excluded_edge_frames": EDGE_FRAMES,
        },
        "boundary_gradient_mean_before_fit": before_edge_mean,
        "boundary_gradient_mean_after_fit": after_edge_mean,
        "higher_boundary_gradient": "before_fit" if before_edge_mean > after_edge_mean else "after_fit",
        "before_fit_signal": {k: v for k, v in before_signal.items()
                              if k not in ("smoothed", "peaks", "valleys")},
        "after_fit_signal": {k: v for k, v in after_signal.items()
                             if k not in ("smoothed", "peaks", "valleys")},
        "diameter_amplitude_change_after_minus_before_mm": (
            after_signal["pulse_amplitude_mean_peak_minus_valley_mm"]
            - before_signal["pulse_amplitude_mean_peak_minus_valley_mm"]
        ),
        "manual_labeled_frame_count": len(manual_metrics),
        "manual_metrics": manual_metrics,
        "manual_label_directory": str(MANUAL_MASK_DIR),
        "manual_metric_note": (
            "Dice and absolute diameter error are computed only for manual binary masks "
            "named frame_XXXX.png placed in manual_masks."
        ),
    }
    (OUTPUT_DIR / "ellipse_fit_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(3, 1, height_ratios=[0.5, 0.1, 0.5], hspace=0.2)
    axes = [fig.add_subplot(grid[0])]
    axes.append(fig.add_subplot(grid[2], sharex=axes[0]))
    legend_ax = fig.add_subplot(grid[1])
    legend_ax.axis("off")
    draw_diameter_curve(
        axes[0], times, before_diameters, before_signal,
        "Cleaned RG + FCM mask", "#e8751a", show_legend=False, 
    )
    draw_diameter_curve(
        axes[1], times, after_diameters, after_signal,
        "Notch Repair (final) mask", "#2876c7", show_legend=False, 
    )
    shared_legend = [
        Line2D([], [], color="#e8751a", lw=3, label="Cleaned RG + FCM smoothed diameter"),
        Line2D([], [], color="#2876c7", lw=3, label="Notch Repair (final) smoothed diameter"),
        Line2D([], [], color="#f6c39f", lw=2, label="Cleaned RG + FCM diameter"),
        Line2D([], [], color="#a9b9c5", lw=1, label="Notch Repair (final) diameter"),
        Line2D([], [], color="#d94b5b", marker="o", linestyle="None",
               markersize=8, label="Peaks"),
        Line2D([], [], color="#2d8b57", marker="o", linestyle="None",
               markersize=8, label="Valleys"),
        Line2D([], [], color="#d94b5b", ls="--", lw=1.2, label=r"$D_{max}$"),
        Line2D([], [], color="#2d8b57", ls="--", lw=1.2, label=r"$D_{min}$"),
    ]
    legend_ax.legend(handles=shared_legend, loc="center", ncol=4,
                     fontsize=10, frameon=False)
    # fig.suptitle("Lumen Diameter with Detected Peaks and Valleys", fontsize=20)
    # Keep only enough inter-panel space for the centered shared legend.
    fig.subplots_adjust(top=0.92, bottom=0.10, hspace=0.1)
    # fig.tight_layout()
    save_figure(
        fig,
        OUTPUT_DIR / "ellipse_fit_diameter_comparison.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)

    save_diameter_curve(
        times,
        before_diameters,
        before_signal,
        "Cleaned RG + FCM mask",
        OUTPUT_DIR / "ellipse_fit_diameter_before_notch_repair.png",
        "#e8751a",
    )
    save_diameter_curve(
        times,
        after_diameters,
        after_signal,
        "Notch Repair (final) mask",
        OUTPUT_DIR / "ellipse_fit_diameter_after_notch_repair.png",
        "#2876c7",
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(times, before_edges, color="#d97706", lw=1.0, label="Before fit")
    ax.plot(times, after_edges, color="#2563eb", lw=1.0, label="After notch repair")
    ax.set(title="Boundary gradient before and after notch repair (proxy)",
           xlabel="Time (s)", ylabel="Mean Sobel magnitude")
    ax.grid(alpha=0.25)
    ax.legend()
  
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "ellipse_fit_boundary_gradient_comparison.png", dpi=180)
    plt.close(fig)

    candidates = {0}
    for signal in (before_signal, after_signal):
        if len(signal["peaks"]):
            candidates.add(int(signal["peaks"][np.argmax(signal["smoothed"][signal["peaks"]])]))
        if len(signal["valleys"]):
            candidates.add(int(signal["valleys"][np.argmin(signal["smoothed"][signal["valleys"]])]))
    with (OUTPUT_DIR / "manual_label_candidate_frames.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "selection_reason"])
        for frame in sorted(candidates):
            reasons = ["frame_0"] if frame == 0 else []
            if frame in set(before_signal["peaks"]) | set(after_signal["peaks"]):
                reasons.append("pulse_peak")
            if frame in set(before_signal["valleys"]) | set(after_signal["valleys"]):
                reasons.append("pulse_valley")
            writer.writerow([frame, ";".join(reasons)])

    fig, axes = plt.subplots(len(candidates), 2,
                             figsize=(6, max(4, 3.2 * len(candidates))), squeeze=False)
    for row_index, frame in enumerate(sorted(candidates)):
        image = read_gray(REFERENCE_DIR / f"frame_{frame:04d}.png")
        for col_index, (label, mask, color) in enumerate((
            ("Before Notch Repair", before_masks[frame], "orange"),
            ("After Notch Repair", after_masks[frame], "cyan"),
        )):
            ax = axes[row_index, col_index]
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            if np.any(mask):
                ax.contour(mask > 0, levels=[0.5], colors=[color], linewidths=1.3)
            ax.text(
                0.03, 0.97, f"Frame {frame}",
                transform=ax.transAxes,
                ha="left", va="top",
                color="white", fontsize=10, fontweight="bold",
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 3},
            )
            # Show each column label once, above the first row of images.
            if row_index == 0:
                ax.set_title(label, fontsize=12)
            ax.axis("off")
    fig.suptitle("Before and After Notch Repair on Aligned Frames",
                 fontsize=14,
                  y=0.98)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "ellipse_fit_mask_comparison_examples.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Ellipse-fit comparison outputs saved in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
