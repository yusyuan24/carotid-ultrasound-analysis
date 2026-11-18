"""Compare contrast enhancement across all current filter outputs.

This script is intentionally not run automatically. Review the input folder
and parameters, then run it yourself when ready.
"""
from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FILTER_RESULTS_DIR = ROOT / "filter_results"
OUTPUT_DIR = ROOT / "contrast_results"

# Current filter configurations produced by filter_comparison.py.
FILTER_METHODS = [
    "mean",
    "median",
    "butterworth_D0.50_order10",
    "butterworth_D0.15_order2",
    "gaussian_k5_sigma0.1",
    "gaussian_k5_sigma1.0",
    "lee_speckle",
]

# imadjust(I) equivalent: saturate the bottom/top 1% and stretch between them.
IMADJUST_LOW_PERCENTILE = 1.0
IMADJUST_HIGH_PERCENTILE = 99.0

# adapthisteq/CLAHE settings: local tile size and contrast clipping limit.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def histogram_entropy(image):
    counts = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def contrast_metrics(image):
    p5, p95 = np.percentile(image, [5, 95])
    return {
        "intensity_std": float(np.std(image)),
        "p95_minus_p5": float(p95 - p5),
        "entropy_bits": histogram_entropy(image),
    }


def imadjust_stretch(image):
    """Approximate MATLAB imadjust(I) for uint8 grayscale images."""
    low, high = np.percentile(
        image, [IMADJUST_LOW_PERCENTILE, IMADJUST_HIGH_PERCENTILE]
    )
    if high <= low:
        return image.copy()
    adjusted = (image.astype(np.float32) - low) / (high - low)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    return np.round(adjusted * 255.0).astype(np.uint8)


def main():
    missing = [name for name in FILTER_METHODS
               if not (FILTER_RESULTS_DIR / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing filter output folders: " + ", ".join(missing)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contrast_methods = [
        "filtered_input",
        "imadjust",
        "histogram_equalization",
        "adapthisteq_clahe",
    ]

    clahe_filter = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )

    sample_indices_by_filter = {}
    sample_images = {}
    midpoint_images = {}
    metric_rows = []

    frame_counts = {}
    for filter_name in FILTER_METHODS:
        input_dir = FILTER_RESULTS_DIR / filter_name
        frame_paths = sorted(input_dir.glob("frame_*.png"))
        if not frame_paths:
            raise FileNotFoundError(f"No PNG frames found in {input_dir}")
        frame_counts[filter_name] = len(frame_paths)
        sample_indices = sorted(set([
            0,
            len(frame_paths) // 2,
            len(frame_paths) - 1,
        ]))
        sample_indices_by_filter[filter_name] = sample_indices
        sample_images[filter_name] = {}
        midpoint = len(frame_paths) // 2

        output_dirs = {
            method: OUTPUT_DIR / filter_name / method
            for method in [
                "imadjust",
                "histogram_equalization",
                "adapthisteq_clahe",
            ]
        }
        for folder in output_dirs.values():
            folder.mkdir(parents=True, exist_ok=True)

        for index, path in enumerate(frame_paths):
            source = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if source is None:
                raise ValueError(f"Could not read image: {path}")

            results = {
                "filtered_input": source,
                "imadjust": imadjust_stretch(source),
                "histogram_equalization": cv2.equalizeHist(source),
                "adapthisteq_clahe": clahe_filter.apply(source),
            }

            for method, image in results.items():
                metric_rows.append({
                    "filter_method": filter_name,
                    "frame": index,
                    "contrast_method": method,
                    **contrast_metrics(image),
                })
                if method in output_dirs:
                    cv2.imwrite(str(output_dirs[method] / path.name), image)

            if index in sample_indices:
                sample_images[filter_name][index] = results
            if index == midpoint:
                midpoint_images[filter_name] = results

    metrics_path = OUTPUT_DIR / "contrast_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as file:
        fields = [
            "filter_method", "frame", "contrast_method",
            "intensity_std", "p95_minus_p5", "entropy_bits",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    # Mean statistics for every filter x contrast-method combination.
    summary_rows = []
    for filter_name in FILTER_METHODS:
        for contrast_method in contrast_methods:
            rows = [
                row for row in metric_rows
                if row["filter_method"] == filter_name
                and row["contrast_method"] == contrast_method
            ]
            summary_rows.append({
                "filter_method": filter_name,
                "contrast_method": contrast_method,
                "frames": len(rows),
                "mean_intensity_std": float(np.mean([
                    r["intensity_std"] for r in rows
                ])),
                "mean_p95_minus_p5": float(np.mean([
                    r["p95_minus_p5"] for r in rows
                ])),
                "mean_entropy_bits": float(np.mean([
                    r["entropy_bits"] for r in rows
                ])),
            })

    with (OUTPUT_DIR / "contrast_metrics_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        fields = [
            "filter_method", "contrast_method", "frames", "mean_intensity_std",
            "mean_p95_minus_p5", "mean_entropy_bits",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # For every filter, show the three contrast methods at frames 0, midpoint, last.
    for filter_name in FILTER_METHODS:
        frames = sample_images[filter_name]
        fig, axes = plt.subplots(
            len(frames), len(contrast_methods),
            figsize=(4 * len(contrast_methods), 3 * len(frames)),
            squeeze=False,
        )
        for row, (frame_index, images) in enumerate(frames.items()):
            for col, method in enumerate(contrast_methods):
                axes[row, col].imshow(images[method], cmap="gray", vmin=0, vmax=255)
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(method.replace("_", " "))
                if col == 0:
                    axes[row, col].set_ylabel(f"Frame {frame_index}")
        fig.suptitle(f"Contrast enhancement: {filter_name}", y=1.002)
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"contrast_comparison_{filter_name}.png",
            dpi=180, bbox_inches="tight"
        )
        plt.close(fig)

    # One combined midpoint figure compares all filter outputs side by side.
    fig, axes = plt.subplots(
        len(FILTER_METHODS), len(contrast_methods),
        figsize=(2 * len(contrast_methods), 2 * len(FILTER_METHODS)),
        squeeze=False,
    )
    for row, filter_name in enumerate(FILTER_METHODS):
        for col, method in enumerate(contrast_methods):
            axes[row, col].imshow(
                midpoint_images[filter_name][method],
                cmap="gray", vmin=0, vmax=255
            )
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(method.replace("_", " "))
            if col == 0:
                axes[row, col].set_ylabel(
                    filter_name.replace("_", " "), rotation=0,
                    ha="right", va="center", labelpad=65
                )
    fig.suptitle("All filter outputs with contrast enhancement (midpoint frame)", y=1.002)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "contrast_comparison_all.png",
        dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    total_frames = sum(frame_counts.values())
    print(f"Processed {total_frames} filter-frame inputs across {len(FILTER_METHODS)} filters.")
    print(f"Comparison images and metrics saved in {OUTPUT_DIR}.")
    print("Higher contrast metrics do not guarantee better lumen segmentation; inspect the images.")


if __name__ == "__main__":
    main()
