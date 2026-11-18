"""Plot the shifts recorded by alignment.py without rerunning alignment."""
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
import pydicom


ROOT = Path(__file__).resolve().parent
SHIFT_CSV = ROOT / "aligned_frames" / "alignment_shifts.csv"
DICOM_PATH = ROOT / "CCA DICOM" / "00000000"
OUTPUT_DIR = ROOT / "analysis_output"
OUTPUT_PNG = OUTPUT_DIR / "alignment_check.png"

# Match the acceptance thresholds in alignment.py.
MAX_ACCEPTED_SHIFT_PX = 10.0
MIN_ACCEPTED_RESPONSE = 0.05


def main():
    if not SHIFT_CSV.exists():
        raise FileNotFoundError(
            f"Alignment log not found: {SHIFT_CSV}. Run alignment.py first."
        )

    with SHIFT_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Alignment log is empty: {SHIFT_CSV}")

    frames = np.array([int(row["frame"]) for row in rows], dtype=int)
    shift_magnitude = np.array(
        [float(row["shift_magnitude_px"]) for row in rows], dtype=float
    )
    response = np.array([float(row["response"]) for row in rows], dtype=float)
    corrected = np.array([
        row["correction_applied"].strip().lower() in {"true", "1", "yes"}
        for row in rows
    ])

    if not DICOM_PATH.exists():
        raise FileNotFoundError(f"DICOM file not found: {DICOM_PATH}")
    dicom = pydicom.dcmread(DICOM_PATH, stop_before_pixels=True)
    frame_time_ms = float(dicom.FrameTime)
    time_s = frames * frame_time_ms / 1000.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (shift_ax, response_ax) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.5]},
    )

    # Upper panel: estimated shifts that alignment.py either applied or rejected.
    shift_ax.plot(time_s, shift_magnitude, color="#8a8a8a", lw=1.0, zorder=1)
    for accepted, color, label in (
        (True, "#276FBF", "Correction applied"),
        (False, "#D1495B", "Left uncorrected"),
    ):
        selected = corrected == accepted
        if np.any(selected):
            shift_ax.scatter(
                time_s[selected], shift_magnitude[selected],
                s=15, color=color, label=label, zorder=2,
            )
    shift_ax.axhline(
        MAX_ACCEPTED_SHIFT_PX, color="#D1495B", ls="--", lw=1.2,
        label=f"Maximum accepted shift ({MAX_ACCEPTED_SHIFT_PX:g} px)",
    )
    shift_ax.set_ylabel("Estimated shift magnitude (pixels)", fontsize=12)
    shift_ax.set_title("Frame shifts estimated by phase correlation", fontsize=14)
    shift_ax.grid(alpha=0.25)
    shift_ax.legend(loc="lower right", fontsize=12)

    # Lower panel: phase-correlation response and its acceptance threshold.
    response_ax.plot(time_s, response, color="#6A4C93", lw=1.0)
    response_ax.scatter(
        time_s[corrected], response[corrected],
        s=12, color="#276FBF", label="Correction applied",
    )
    if np.any(~corrected):
        response_ax.scatter(
            time_s[~corrected], response[~corrected],
            s=18, color="#D1495B", label="Left uncorrected",
        )
    response_ax.axhline(
        MIN_ACCEPTED_RESPONSE, color="#D1495B", ls="--", lw=1.2,
        label=f"Minimum accepted response ({MIN_ACCEPTED_RESPONSE:g})",
    )
    response_ax.set_xlabel("Time (s)", fontsize=12)
    response_ax.set_ylabel("Phase-correlation response", fontsize=12)
    response_ax.grid(alpha=0.25)
    response_ax.legend(loc="lower right", fontsize=12)

    corrected_count = int(np.count_nonzero(corrected))
    summary = (
        f"{len(frames)} frames | corrected: {corrected_count} | "
        f"mean shift: {np.mean(shift_magnitude):.2f} px | "
        f"maximum shift: {np.max(shift_magnitude):.2f} px"
    )
    
    fig.suptitle(
        "Alignment Diagnostics Relative to the First Cropped Frame",
        fontsize=16,
    )
    fig.text(
        x=0.5,     
        y=0.91,     
        s=summary, 
        ha="center",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Alignment check plot saved to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
