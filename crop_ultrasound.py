"""Crop the actual sonographic viewport from each DICOM cine frame."""
from pathlib import Path
import json
import pydicom
from PIL import Image

ROOT = Path(__file__).resolve().parent
# File path to the DICOM file.
SOURCE = ROOT / 'CCA DICOM' / '00000000'
OUT = ROOT / 'cropped_frames'
OUT.mkdir(exist_ok=True)

ds = pydicom.dcmread(SOURCE)
frames = ds.pixel_array
# Crop the B-mode image, trimming the display header, settings panel, controls,
# color marker at the top-left, and depth ruler at the right edge.
x0, x1, y0, y1 = 450, 650, 200, 400  # half-open pixel bounds
for i, frame in enumerate(frames):
    output = OUT / f'frame_{i:04d}.png'
    Image.fromarray(frame[y0:y1, x0:x1]).save(output)

metadata = {
    'source': str(SOURCE.relative_to(ROOT)),
    'frames': int(len(frames)),
    'crop_xyxy_exclusive': [x0, y0, x1, y1],
    'crop_size_px': [x1-x0, y1-y0],
    'pixel_spacing_mm': abs(float(ds.SequenceOfUltrasoundRegions[0].PhysicalDeltaY))*10,
    'frame_time_ms': float(ds.FrameTime),
    'note': 'Crop retains the B-mode scan area and removes the screen header, settings panel, playback controls, top-left marker, and right-side depth ruler.'
}
(OUT / 'crop_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print(json.dumps(metadata, indent=2))
