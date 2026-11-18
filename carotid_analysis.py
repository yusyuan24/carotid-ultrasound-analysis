"""Measure lumen diameter using the quality-screened fitted-ellipse masks."""
from pathlib import Path
import csv
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter


ROOT = Path(__file__).resolve().parent
FILTER_METHOD = "gaussian_k5_sigma1.0"
CONTRAST_METHOD = "histogram_equalization"
INPUT_DIR = ROOT / "contrast_results" / FILTER_METHOD / CONTRAST_METHOD
OUTPUT_DIR = ROOT / "analysis_output"
SEGMENTATION_DIR = ROOT / "segmentation_rg_fcm_ellipse_results"
FINAL_MASK_DIR = SEGMENTATION_DIR / "all_frame_masks"
SEGMENTATION_METRICS_CSV = SEGMENTATION_DIR / "segmentation_metrics.csv"

MM_PER_PIXEL = 0.062397

# Cine signal analysis settings.
SYSTOLIC_PRESSURE_MMHG = 105.0
DIASTOLIC_PRESSURE_MMHG = 70.0
MIN_BEAT_PERIOD_S = 0.6
BEAT_PEAK_PROMINENCE_MM = 0.05
EDGE_FRAMES = 15
QC_FRAME_INDICES = (0, 125, 250, 375, 499)


def read_frame_time_ms():
    try:
        import pydicom
        dicom_path = ROOT / "CCA DICOM" / "00000000"
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        return float(ds.FrameTime)
    except Exception as exc:
        raise RuntimeError(
            "Could not read FrameTime from CCA DICOM/00000000; "
            "install pydicom and check the DICOM path."
        ) from exc


def detect_beats(smoothed, frame_ms, min_period_s=MIN_BEAT_PERIOD_S,
                 edge_frames=EDGE_FRAMES):
    """Detect systolic peaks, then one valley between each adjacent pair."""
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


def main():
    frame_paths = sorted(INPUT_DIR.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No preprocessed frames found in {INPUT_DIR}")
    if not SEGMENTATION_METRICS_CSV.is_file():
        raise FileNotFoundError(
            f"Segmentation quality CSV not found: {SEGMENTATION_METRICS_CSV}. "
            "Run segmentation_rg_fcm_ellipse.py first."
        )
    with SEGMENTATION_METRICS_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        quality_by_frame = {
            int(row["frame"]): row for row in csv.DictReader(file)
        }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_time_ms = read_frame_time_ms()
    measurements = []
    qc_images = {}
    qc_indices = set(QC_FRAME_INDICES)

    for frame_index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")
        if image.shape != (200, 200):
            raise ValueError(f"Expected 200x200 input in {frame_path}, got {image.shape}")

        mask_path = FINAL_MASK_DIR / f"final_ellipse_mask_{frame_index:04d}.png"
        final_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if final_mask is None:
            raise FileNotFoundError(f"Could not read final segmentation mask: {mask_path}")
        if final_mask.shape != image.shape:
            raise ValueError(f"Mask shape {final_mask.shape} does not match frame {image.shape}: {mask_path}")
        if frame_index not in quality_by_frame:
            raise KeyError(f"Frame {frame_index} is missing from {SEGMENTATION_METRICS_CSV}")
        quality = quality_by_frame[frame_index]
        ellipse_mask = final_mask > 0
        area_px = int(np.count_nonzero(ellipse_mask))
        candidate_diameter_px = float(np.sqrt(4.0 * area_px / np.pi)) if area_px else np.nan
        candidate_diameter_mm = candidate_diameter_px * MM_PER_PIXEL
        usable = quality.get("diameter_usable", "False").strip().lower() in ("true", "1", "yes")
        try:
            ellipse_iou = float(quality["cleaned_ellipse_iou"])
        except (KeyError, TypeError, ValueError):
            ellipse_iou = np.nan
        usable = bool(usable and np.isfinite(ellipse_iou) and ellipse_iou >= 0.85)
        valid = bool(usable and np.isfinite(candidate_diameter_mm))
        diameter_px = candidate_diameter_px if valid else np.nan
        diameter_mm = candidate_diameter_mm if valid else np.nan
        time_s = frame_index * frame_time_ms / 1000.0
        measurements.append({
            "frame": frame_index,
            "time_s": time_s,
            "area_px": area_px,
            "diameter_candidate_px": candidate_diameter_px,
            "diameter_candidate_mm": candidate_diameter_mm,
            "diameter_raw_px": diameter_px,
            "diameter_raw_mm": diameter_mm,
            "valid": valid,
            "quality_flag": quality.get("quality_flag", ""),
            "quality_reason": quality.get("quality_reason", ""),
            "ellipse_inlier_ratio": quality.get("ellipse_inlier_ratio", quality.get("inlier_ratio", "")),
            "cleaned_ellipse_iou": ellipse_iou,
        })

        if frame_index in qc_indices:
            qc_images[frame_index] = (image.copy(), final_mask.copy(), valid)

    frame_numbers = np.array([m["frame"] for m in measurements], dtype=int)
    time_s = np.array([m["time_s"] for m in measurements], dtype=float)
    raw_mm = np.array([m["diameter_raw_mm"] for m in measurements], dtype=float)
    valid = np.isfinite(raw_mm)
    if np.count_nonzero(valid) < 10:
        raise RuntimeError(
            f"Only {np.count_nonzero(valid)} valid measurements; inspect segmentation QC."
        )

    # Interpolate only invalid frames, then smooth pixel-scale variation.
    filled_mm = np.interp(frame_numbers, frame_numbers[valid], raw_mm[valid])
    window = min(31, len(filled_mm) if len(filled_mm) % 2 else len(filled_mm) - 1)
    if window < 5:
        raise RuntimeError("Too few frames for the selected Savitzky-Golay smoothing.")
    smoothed_filled_mm = savgol_filter(filled_mm, window_length=window, polyorder=3)
    # Keep rejected frames as gaps in reported curves; interpolation is used only
    # internally to let the smoother bridge short missing segments.
    smoothed_mm = smoothed_filled_mm.copy()
    smoothed_mm[~valid] = np.nan

    peaks, valleys = detect_beats(smoothed_filled_mm, frame_time_ms)
    peaks = peaks[valid[peaks]]
    valleys = valleys[valid[valleys]]
    mean_peak = float(np.mean(smoothed_filled_mm[peaks])) if len(peaks) else float("nan")
    mean_valley = float(np.mean(smoothed_filled_mm[valleys])) if len(valleys) else float("nan")
    log_diameter_ratio = (
        float(np.log(mean_peak / mean_valley))
        if len(peaks) and len(valleys) and mean_valley > 0 else float("nan")
    )
    relative_diameter_change = (
        float((mean_peak - mean_valley) / mean_valley)
        if len(peaks) and len(valleys) and mean_valley > 0 else float("nan")
    )
    mean_peak_interval_frames = (
        float(np.mean(np.diff(peaks))) if len(peaks) > 1 else float("nan")
    )
    pulse_rate_bpm = (
        float(60_000.0 / (mean_peak_interval_frames * frame_time_ms))
        if np.isfinite(mean_peak_interval_frames) and mean_peak_interval_frames > 0
        else float("nan")
    )

    # Inverse map from diameter to pressure: maximum diameter -> diastolic
    # pressure; minimum diameter -> systolic pressure.
    min_diameter = float(np.min(smoothed_filled_mm[valid]))
    max_diameter = float(np.max(smoothed_filled_mm[valid]))
    diameter_range = max_diameter - min_diameter
    if np.isfinite(diameter_range) and diameter_range > 0:
        relative_diameter_position = np.clip(
            (smoothed_filled_mm - min_diameter) / (max_diameter - min_diameter),
            0.0,
            1.0,
        )
        estimated_pressure_mmHg = (
            DIASTOLIC_PRESSURE_MMHG
            + relative_diameter_position
            * (SYSTOLIC_PRESSURE_MMHG - DIASTOLIC_PRESSURE_MMHG)
        )
    else:
        relative_diameter_position = np.full_like(smoothed_filled_mm, np.nan)
        estimated_pressure_mmHg = np.full_like(smoothed_filled_mm, np.nan)
    relative_diameter_position[~valid] = np.nan
    estimated_pressure_mmHg[~valid] = np.nan

    peak_set, valley_set = set(peaks.tolist()), set(valleys.tolist())
    with (OUTPUT_DIR / "diameter_timeseries.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", "time_s", "lumen_area_px", "diameter_candidate_px",
            "diameter_candidate_mm", "diameter_raw_px", "diameter_raw_mm",
            "diameter_smoothed_mm", "ellipse_inlier_ratio", "cleaned_ellipse_iou",
            "quality_flag", "quality_reason", "valid", "event",
        ])
        for m, smooth in zip(measurements, smoothed_mm):
            i = m["frame"]
            event = "peak" if i in peak_set else ("valley" if i in valley_set else "")
            writer.writerow([
                i, m["time_s"], m["area_px"], m["diameter_candidate_px"],
                m["diameter_candidate_mm"], m["diameter_raw_px"],
                m["diameter_raw_mm"], smooth, m["ellipse_inlier_ratio"],
                m["cleaned_ellipse_iou"], m["quality_flag"],
                m["quality_reason"], m["valid"], event,
            ])

    with (OUTPUT_DIR / "peak_valley_events.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["type", "frame", "time_s", "diameter_mm"])
        for kind, indices in (("peak", peaks), ("valley", valleys)):
            for i in indices:
                writer.writerow([kind, int(i), float(time_s[i]), float(smoothed_filled_mm[i])])

    with (OUTPUT_DIR / "estimated_pressure_timeseries.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", "time_s", "diameter_smoothed_mm", "valid",
            "relative_diameter_position", "estimated_pressure_mmHg",
        ])
        for i in range(len(frame_numbers)):
            writer.writerow([
                int(frame_numbers[i]), float(time_s[i]), smoothed_mm[i], bool(valid[i]),
                float(relative_diameter_position[i]),
                float(estimated_pressure_mmHg[i]),
            ])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_s, raw_mm, color="#9bb7ca", lw=0.8, label="Measured diameter")
    ax.plot(time_s, smoothed_mm, color="#124e78", lw=1.6, label="Smoothed diameter")
    ax.scatter(time_s[peaks], smoothed_mm[peaks], color="#d94b4b", s=108, label="Peaks")
    ax.scatter(time_s[valleys], smoothed_mm[valleys], color="#2a9d65", s=108, label="Valleys")

    ax.set_title("Lumen Diameter", fontsize=18)
    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_ylabel("Lumen diameter (mm)", fontsize=16)

    ax.grid(alpha=0.25)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diameter_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_s, estimated_pressure_mmHg, color="#7b3294", lw=1.6,
            label="Estimated pressure")
    ax.axhline(SYSTOLIC_PRESSURE_MMHG, color="#d94b4b", ls="--", lw=1,
               label=f"Systolic reference ({SYSTOLIC_PRESSURE_MMHG:.0f} mmHg)")
    ax.axhline(DIASTOLIC_PRESSURE_MMHG, color="#777777", ls="--", lw=1,
               label=f"Diastolic reference ({DIASTOLIC_PRESSURE_MMHG:.0f} mmHg)")
    # Find valleys on the pressure curve itself. Do not infer them from the
    # diameter peak indices: the green guide should pass through an actual
    # local minimum of the plotted pressure signal.
    pressure_valid = np.isfinite(estimated_pressure_mmHg)
    pressure_for_detection = np.interp(
        frame_numbers,
        frame_numbers[pressure_valid],
        estimated_pressure_mmHg[pressure_valid],
    )
    pressure_trough_indices, _ = find_peaks(
        -pressure_for_detection,
        distance=max(1, int(MIN_BEAT_PERIOD_S * 1000.0 / frame_time_ms)),
        prominence=BEAT_PEAK_PROMINENCE_MM,
    )
    pressure_trough_indices = pressure_trough_indices[
        (pressure_trough_indices >= EDGE_FRAMES)
        & (pressure_trough_indices < len(frame_numbers) - EDGE_FRAMES)
        & pressure_valid[pressure_trough_indices]
    ]
    if len(pressure_trough_indices):
        lowest_cycle_trough = float(
            np.min(estimated_pressure_mmHg[pressure_trough_indices])
        )
        ax.axhline(
            lowest_cycle_trough, color="#2a9d65", ls="--", lw=1.2,
            label=f"Lowest cycle trough ({lowest_cycle_trough:.1f} mmHg)",
        )
    ax.set_title("Estimated Pressure Curve", fontsize=18)
    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_ylabel("Estimated Pressure (mmHg)", fontsize=16)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "estimated_pressure_curve.png", dpi=180)
    plt.close(fig)

    if qc_images:
        fig, axes = plt.subplots(1, len(qc_images), figsize=(15, 3.6))
        axes = np.atleast_1d(axes)
        for ax, (idx, (image, mask, usable)) in zip(axes, qc_images.items()):
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.contour(mask > 0, levels=[0.5], colors=["cyan"], linewidths=0.9)
            ax.set_title(f"Frame {idx} ({'PASS' if usable else 'REVIEW'})")
            ax.axis("off")
        fig.suptitle("Final fitted-ellipse segmentation QC (cyan = selected lumen)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "segmentation_qc.png", dpi=180)
        plt.close(fig)

    pressure_log_ratio = float(np.log(SYSTOLIC_PRESSURE_MMHG / DIASTOLIC_PRESSURE_MMHG))
    summary = {
        "frames": int(len(frame_paths)),
        "frame_time_ms": frame_time_ms,
        "duration_s": float(time_s[-1]),
        "pixel_spacing_mm": MM_PER_PIXEL,
        "valid_measurements": int(np.count_nonzero(valid)),
        "quality_rejected_frames": int(np.count_nonzero(~valid)),
        "diameter_mask_source": str(FINAL_MASK_DIR),
        "quality_metrics_source": str(SEGMENTATION_METRICS_CSV),
        "diameter_definition": "Equivalent diameter from final ellipse-mask area: sqrt(4A/pi).",
        "valid_fraction": float(np.mean(valid)),
        "peak_count": int(len(peaks)),
        "valley_count": int(len(valleys)),
        "detected_cycle_count": int(min(len(peaks), len(valleys))),
        "mean_peak_interval_frames": mean_peak_interval_frames,
        "estimated_pulse_rate_bpm": pulse_rate_bpm,
        "mean_peak_diameter_mm": mean_peak,
        "mean_valley_diameter_mm": mean_valley,
        "estimated_pressure_min_mmHg": (
            float(np.nanmin(estimated_pressure_mmHg))
            if np.any(np.isfinite(estimated_pressure_mmHg)) else None
        ),
        "estimated_pressure_max_mmHg": (
            float(np.nanmax(estimated_pressure_mmHg))
            if np.any(np.isfinite(estimated_pressure_mmHg)) else None
        ),
        "log_peak_over_valley_diameter": log_diameter_ratio,
        "relative_diameter_change": relative_diameter_change,
        "systolic_pressure_mmHg": SYSTOLIC_PRESSURE_MMHG,
        "diastolic_pressure_mmHg": DIASTOLIC_PRESSURE_MMHG,
        "log_systolic_over_diastolic_pressure": pressure_log_ratio,
        "beat_detection": {
            "method": "peaks first; one minimum valley between adjacent peaks",
            "minimum_period_s": MIN_BEAT_PERIOD_S,
            "peak_prominence_mm": BEAT_PEAK_PROMINENCE_MM,
            "excluded_edge_frames": EDGE_FRAMES,
        },
        "filter": FILTER_METHOD,
        "contrast_enhancement": CONTRAST_METHOD,
        "quality_iou_threshold": 0.85,
        "pressure_calibration_min_diameter_mm": min_diameter,
        "pressure_calibration_max_diameter_mm": max_diameter,
        "note": (
            "Uses the trimmed fitted-ellipse masks produced by segmentation_rg_fcm_ellipse.py. "
            "Frames with segmentation quality flag REVIEW, including cleaned-mask/ellipse IoU "
            "below 0.85, are excluded from diameter curves and pulse event statistics. "
            "Estimated pressure follows the specified inverse linear mapping: maximum lumen "
            "diameter maps to diastolic pressure and minimum diameter maps to systolic pressure. "
            "It is an approximation anchored to cuff values that may be from a different date, "
            "not a direct or clinically validated pressure waveform."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Diameter and pulse-analysis outputs saved in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
