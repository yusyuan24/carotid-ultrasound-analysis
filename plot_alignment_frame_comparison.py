"""Create sampled before/after alignment comparisons from saved frame folders."""
from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
BEFORE_DIR = ROOT / "cropped_frames"
AFTER_DIR = ROOT / "aligned_frames"
SHIFT_CSV = AFTER_DIR / "alignment_shifts.csv"
OUTPUT_DIR = ROOT / "analysis_output"
OUTPUT_PNG = OUTPUT_DIR / "alignment_frame_comparison.png"

# ROI used by alignment.py to estimate global translation.
ALIGNMENT_ROI_XYWH = (5, 5, 190, 50)


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read frame: {path}")
    return image


def main():
    before_paths = sorted(BEFORE_DIR.glob("frame_*.png"))
    if not before_paths:
        raise FileNotFoundError(f"No cropped frames found in {BEFORE_DIR}")
    if not SHIFT_CSV.exists():
        raise FileNotFoundError(
            f"Alignment log not found: {SHIFT_CSV}. Run alignment.py first."
        )

    with SHIFT_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        shift_rows = list(csv.DictReader(file))
    if not shift_rows:
        raise ValueError(f"Alignment log is empty: {SHIFT_CSV}")

    shift_by_frame = {int(row["frame"]): row for row in shift_rows}
    max_shift_frame = max(
        shift_rows, key=lambda row: float(row["shift_magnitude_px"])
    )
    n_frames = len(before_paths)
    max_shift_index = int(max_shift_frame["frame"])
    if not 0 <= max_shift_index < n_frames:
        raise ValueError(
            f"Maximum-shift frame index {max_shift_index} is outside the "
            f"available range 0..{n_frames - 1}."
        )
    sample_indices = [max_shift_index]

    fig, axes = plt.subplots(
        len(sample_indices), 3,
        figsize=(10, 4.5 * len(sample_indices)),
        squeeze=False,
    )
    titles = [
        "Before alignment\n(cropped)",
        "After alignment",
        "Overlay\n(before: red, after: green)",
    ]

    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11)

    roi_x, roi_y, roi_w, roi_h = ALIGNMENT_ROI_XYWH
    for row_index, frame_index in enumerate(sample_indices):
        frame_name = f"frame_{frame_index:04d}.png"
        before_path = BEFORE_DIR / frame_name
        after_path = AFTER_DIR / frame_name
        if not after_path.exists():
            raise FileNotFoundError(f"Aligned frame not found: {after_path}")

        before = read_gray(before_path)
        after = read_gray(after_path)
        if before.shape != after.shape:
            raise ValueError(f"Before/after shape mismatch for {frame_name}")

        overlay = np.zeros((*before.shape, 3), dtype=np.uint8)
        overlay[:, :, 0] = before  # red
        overlay[:, :, 1] = after   # green
        overlay[:, :, 2] = before  # blue, so the original is magenta

        image_panels = (before, after, overlay)
        for col, panel in enumerate(image_panels):
            ax = axes[row_index, col]
            if panel.ndim == 2:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(panel)
            ax.add_patch(Rectangle(
                (roi_x, roi_y), roi_w, roi_h,
                fill=False, edgecolor="yellow", linewidth=0.8,
            ))
            ax.axis("off")

        log_row = shift_by_frame.get(frame_index)
        if log_row:
            magnitude = float(log_row["shift_magnitude_px"])
            applied = log_row["correction_applied"].strip().lower()
            row_label = (
                f"Frame {frame_index} | estimated shift {magnitude:.2f} px | "
                f"correction applied: {applied}"
            )
        else:
            row_label = f"Frame {frame_index}"
        axes[row_index, 0].set_ylabel(
            row_label, rotation=0, ha="right", va="center", labelpad=95,
            fontsize=14,
        )

    panel_left = 0.01
    panel_right = 0.99
    fig.suptitle(
        "Alignment Frame Comparison",
        x=(panel_left + panel_right) / 2,
        ha="center",
        fontsize=16,
    )
    fig.text(
        0.5, 0.025,
        "Yellow box indicates the phase-correlation ROI used for alignment.",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    fig.subplots_adjust(left=panel_left, right=panel_right, top=0.91, bottom=0.08,
                        wspace=0.08, hspace=0.12)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Frame comparison saved to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
