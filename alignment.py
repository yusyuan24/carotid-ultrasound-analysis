from pathlib import Path
import csv

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "cropped_frames"
OUTPUT_DIR = ROOT / "aligned_frames"
OUTPUT_DIR.mkdir(exist_ok=True)

# Read the already-cropped frames. Filenames should be frame_0000.png, etc.
frame_paths = sorted(INPUT_DIR.glob("frame_*.png"))
if not frame_paths:
    raise FileNotFoundError(f"No frame_*.png files found in {INPUT_DIR}")

frames = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in frame_paths]
if any(frame is None for frame in frames):
    raise ValueError("At least one cropped frame could not be read.")
frame_shape = frames[0].shape
if any(frame.shape != frame_shape for frame in frames):
    raise ValueError("All cropped frames must have the same dimensions.")

# Estimate translation from tissue above the vessel in the cropped frame.
# These coordinates are relative to each cropped image.
rx0, rx1 = 5, 195
ry0, ry1 = 5, 55
if frame_shape[1] < rx1 or frame_shape[0] < ry1:
    raise ValueError(
        f"Cropped frames are too small for the alignment ROI: {frame_shape}."
    )

def to_gray(frame):
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

reference_gray = to_gray(frames[0]).astype(np.float32)
reference_roi = reference_gray[ry0:ry1, rx0:rx1]
hann = cv2.createHanningWindow(
    (reference_roi.shape[1], reference_roi.shape[0]),
    cv2.CV_32F
)

rows = []

for i, frame in enumerate(frames):
    gray = to_gray(frame).astype(np.float32)
    moving_roi = gray[ry0:ry1, rx0:rx1]

    # phaseCorrelate 回傳 moving ROI 相對 reference ROI 的位移 (dx, dy)
    (dx, dy), response = cv2.phaseCorrelate(
        reference_roi, moving_roi, hann
    )
    shift_size = float(np.hypot(dx, dy))

    # 低 response 或極大位移可能是錯誤估計；這些影格不套用校正
    reliable = response >= 0.05 and shift_size <= 10

    if reliable:
        # moving 影格要往反方向平移，才能對齊 reference
        matrix = np.float32([
            [1, 0, -dx],
            [0, 1, -dy]
        ])
        aligned = cv2.warpAffine(
            frame,
            matrix,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
    else:
        aligned = frame.copy()

    cv2.imwrite(
        str(OUTPUT_DIR / f"frame_{i:04d}.png"),
        aligned
    )

    rows.append([i, dx, dy, shift_size, response, reliable])

# 儲存每幀估計的位移與可信度
with open(OUTPUT_DIR / "alignment_shifts.csv",
          "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame", "shift_x_px", "shift_y_px",
        "shift_magnitude_px", "response", "correction_applied"
    ])
    writer.writerows(rows)

print(f"完成：{len(frames)} 幀對位結果已存入 {OUTPUT_DIR}")
