"""Compare filter types and parameter sets on aligned carotid cine frames.

This script is intentionally not run automatically. Review/edit parameters,
then run it yourself when ready.
"""
from pathlib import Path
import csv


import cv2
import numpy as np
import matplotlib.pyplot as plt
import textwrap


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "aligned_frames"
OUTPUT_DIR = ROOT / "filter_results"

# Filter parameters; keep them identical across frames for a fair comparison.
MEAN_KERNEL = 3
MEDIAN_KERNEL = 3
# Two parameter sets for each filter requested for comparison.
GAUSSIAN_CONFIGS = {
    "gaussian_k5_sigma0.1": (5, 0.1),
    "gaussian_k5_sigma1.0": (5, 1.0),
}
BUTTERWORTH_CONFIGS = {
    "butterworth_D0.50_order10": (0.50, 10),
    "butterworth_D0.15_order2": (0.15, 2),
}
LEE_KERNEL = 3


def butterworth_lowpass(image_u8, cutoff_cycles_per_pixel, order):
    """Apply a Butterworth low-pass filter in the frequency domain."""
    image = image_u8.astype(np.float32) / 255.0
    height, width = image.shape
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    grid_x, grid_y = np.meshgrid(fx, fy)
    radius = np.sqrt(grid_x**2 + grid_y**2)
    transfer = 1.0 / (
        1.0 + (radius / cutoff_cycles_per_pixel) ** (2 * order)
    )
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * transfer)).real
    return np.clip(np.round(filtered * 255.0), 0, 255).astype(np.uint8)


def lee_speckle_filter(image_u8):
    """Simplified adaptive Lee filter for multiplicative ultrasound speckle."""
    image = image_u8.astype(np.float32) / 255.0
    size = (LEE_KERNEL, LEE_KERNEL)
    local_mean = cv2.boxFilter(image, -1, size, normalize=True)
    local_mean_sq = cv2.boxFilter(image * image, -1, size, normalize=True)
    local_var = np.maximum(local_mean_sq - local_mean * local_mean, 0.0)

    # Estimate the speckle coefficient of variation squared from the lower
    # 20th percentile of local normalized variances (relatively homogeneous areas).
    normalized_var = local_var / (local_mean * local_mean + 1e-6)
    noise_cv_sq = float(np.percentile(normalized_var, 20))
    noise_var = noise_cv_sq * local_mean * local_mean
    weight = local_var / (local_var + noise_var + 1e-6)
    result = local_mean + weight * (image - local_mean)
    return np.clip(np.round(result * 255.0), 0, 255).astype(np.uint8)


def make_filtered_images(image):
    results = {
        "mean": cv2.blur(image, (MEAN_KERNEL, MEAN_KERNEL)),
        "median": cv2.medianBlur(image, MEDIAN_KERNEL),
        "lee_speckle": lee_speckle_filter(image),
    }
    for name, (cutoff, order) in BUTTERWORTH_CONFIGS.items():
        results[name] = butterworth_lowpass(image, cutoff, order)
    for name, (kernel, sigma) in GAUSSIAN_CONFIGS.items():
        results[name] = cv2.GaussianBlur(
            image, (kernel, kernel), sigmaX=sigma
        )
    return results


def image_metrics(original, filtered):
    """Exploratory smoothness/detail measures, not ground-truth quality scores."""
    original_f = original.astype(np.float32)
    filtered_f = filtered.astype(np.float32)
    gx = cv2.Sobel(filtered_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered_f, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "residual_std": float(np.std(original_f - filtered_f)),
        "mean_gradient_magnitude": float(np.mean(np.hypot(gx, gy))),
        "laplacian_variance": float(cv2.Laplacian(filtered_f, cv2.CV_32F).var()),
    }


def main():
    frame_paths = sorted(INPUT_DIR.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(
            f"No aligned PNG frames found in {INPUT_DIR}. Run alignment first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    method_names = [
        "mean",
        "median",
        *BUTTERWORTH_CONFIGS.keys(),
        *GAUSSIAN_CONFIGS.keys(),
        "lee_speckle",
    ]
    method_dirs = {name: OUTPUT_DIR / name for name in method_names}
    for folder in method_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    original_metric_rows = []
    comparison_frames = {}
    sample_indices = sorted(set([
        0,
        len(frame_paths) // 2,
        len(frame_paths) - 1,
    ]))

    for index, path in enumerate(frame_paths):
        original = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"Could not read image: {path}")

        results = make_filtered_images(original)
        original_metric_rows.append({
            "frame": index,
            "method": "original",
            **image_metrics(original, original),
        })
        for method, filtered in results.items():
            cv2.imwrite(
                str(method_dirs[method] / path.name),
                filtered,
            )
            metric_rows.append({
                "frame": index,
                "method": method,
                **image_metrics(original, filtered),
            })

        if index in sample_indices:
            comparison_frames[index] = (original, results)

    # Include original-image measurements as a baseline in the frame-level CSV.
    metric_rows = original_metric_rows + metric_rows
    metrics_path = OUTPUT_DIR / "filter_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "frame", "method", "residual_std",
            "mean_gradient_magnitude", "laplacian_variance",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    # Summarize all frames by method and compare detail measures with the original.
    summary_rows = []
    for method in ["original", *method_names]:
        method_rows = [row for row in metric_rows if row["method"] == method]
        summary_rows.append({
            "method": method,
            "frames": len(method_rows),
            "residual_std_mean": float(np.mean([
                row["residual_std"] for row in method_rows
            ])),
            "mean_gradient_magnitude": float(np.mean([
                row["mean_gradient_magnitude"] for row in method_rows
            ])),
            "gradient_change_percent_vs_original": 0.0,
            "laplacian_variance": float(np.mean([
                row["laplacian_variance"] for row in method_rows
            ])),
            "laplacian_change_percent_vs_original": 0.0,
        })

    original_summary = summary_rows[0]
    for row in summary_rows[1:]:
        row["gradient_change_percent_vs_original"] = (
            100.0 * (row["mean_gradient_magnitude"]
                     - original_summary["mean_gradient_magnitude"])
            / max(original_summary["mean_gradient_magnitude"], 1e-9)
        )
        row["laplacian_change_percent_vs_original"] = (
            100.0 * (row["laplacian_variance"]
                     - original_summary["laplacian_variance"])
            / max(original_summary["laplacian_variance"], 1e-9)
        )

    summary_fields = [
        "method", "frames", "residual_std_mean",
        "mean_gradient_magnitude", "gradient_change_percent_vs_original",
        "laplacian_variance", "laplacian_change_percent_vs_original",
    ]
    summary_path = OUTPUT_DIR / "filter_metrics_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Generate a concise interpretation while explicitly avoiding quality claims
    # that these no-reference metrics cannot support.
    filtered_summaries = summary_rows[1:]
    closest_to_input = min(filtered_summaries,
                           key=lambda row: row["residual_std_mean"])
    lowest_gradient = min(filtered_summaries,
                          key=lambda row: row["mean_gradient_magnitude"])
    highest_laplacian = max(filtered_summaries,
                            key=lambda row: row["laplacian_variance"])
    lines = [
        "Filter comparison analysis (averaged over all frames)",
        "",
        f"Original mean gradient: {original_summary['mean_gradient_magnitude']:.3f}",
        f"Original Laplacian variance: {original_summary['laplacian_variance']:.3f}",
        f"Smallest residual_std_mean (closest numerically to input): "
        f"{closest_to_input['method']} "
        f"({closest_to_input['residual_std_mean']:.3f}).",
        f"Lowest mean gradient (strongest smoothing proxy): "
        f"{lowest_gradient['method']} "
        f"({lowest_gradient['mean_gradient_magnitude']:.3f}).",
        f"Highest Laplacian variance (most high-frequency variation): "
        f"{highest_laplacian['method']} "
        f"({highest_laplacian['laplacian_variance']:.3f}).",
        "",
        "Interpretation limits:",
        "- residual_std_mean is the standard deviation of original minus filtered image;",
        "  it measures image change, not the amount of noise remaining.",
        "- Gradient and Laplacian measures respond to both anatomical edges and noise.",
        "- These no-reference metrics cannot identify the best filter by themselves.",
        "  Inspect the comparison figure and confirm that the lumen boundary is preserved.",
    ]
    (OUTPUT_DIR / "filter_analysis.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"Processed {len(frame_paths)} frames.")
    print("See filter_metrics_summary.csv and filter_analysis.txt for the summary.")


if __name__ == "__main__":
    main()