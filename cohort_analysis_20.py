"""Descriptive, paired analysis of the 20 supplied dental images.

The script keeps the segmentation and thresholds used in new_analyse2.ipynb.
It reports candidate image regions, never clinically confirmed lesions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology

import dantistUtils as du


def tooth_mask(image_bgr: np.ndarray) -> np.ndarray:
    _, _, contours, _ = du.find_conturs(image_bgr)
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, -1)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=500)
    return ndimage.binary_fill_holes(mask)


def retained_components(raw: np.ndarray) -> tuple[int, int]:
    retained = morphology.remove_small_objects(raw, min_size=20)
    _, count = ndimage.label(retained)
    return int(count), int(retained.sum())


def analyze(path: Path) -> dict[str, object]:
    row: dict[str, object] = {"image_id": path.stem, "status": ""}
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        row["status"] = "unreadable_image"
        return row

    mask = tooth_mask(image)
    area = int(mask.sum())
    row["mask_pixels"] = area
    if area == 0:
        row["status"] = "segmentation_failed"
        return row

    # Green-channel fluorescence proxy, matching the manuscript's Eq. (1).
    green = image[:, :, 1].astype(np.float32)
    reference_green = float(np.percentile(green[mask], 95))
    if reference_green <= 0:
        row["status"] = "invalid_green_reference"
        return row
    delta_f = np.zeros(mask.shape, dtype=np.float32)
    delta_f[mask] = np.clip((1 - green[mask] / reference_green) * 100, 0, 100)
    green_regions, green_pixels = retained_components((delta_f > 10) & mask)

    # Hybrid dantistUtils score: Δ₁ = ΔG + ΔRloss. Reference means are
    # calculated per B/G/R channel from the same valid masked pixels.
    valid = mask & np.all(image > 0, axis=2)
    if not valid.any():
        row["status"] = "invalid_hybrid_reference"
        return row
    reference_bgr = image[valid].astype(np.float32).mean(axis=0)
    bgr = image.astype(np.float32)
    processed = valid & ((bgr[:, :, 0] <= reference_bgr[0]) | (bgr[:, :, 1] <= reference_bgr[1]))
    delta_g = np.maximum(0, reference_bgr[1] - bgr[:, :, 1])
    delta_r_loss = np.maximum(0, reference_bgr[2] - bgr[:, :, 2])
    delta_1 = delta_g + delta_r_loss
    hybrid_regions, hybrid_pixels = retained_components((delta_1 > 45) & processed)

    row.update(
        status="ok",
        green_reference=round(reference_green, 3),
        green_regions=green_regions,
        green_pixels=green_pixels,
        green_mask_pct=round(100 * green_pixels / area, 4),
        hybrid_processed_pixels=int(processed.sum()),
        hybrid_regions=hybrid_regions,
        hybrid_pixels=hybrid_pixels,
        hybrid_mask_pct=round(100 * hybrid_pixels / area, 4),
        hybrid_q0=round(float(((delta_1 > 45) & processed).sum() / processed.sum()), 6)
        if processed.any() else 0,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.images.glob("QLF-IMG-*"))
    if len(paths) != 20:
        raise ValueError(f"Expected 20 image files, found {len(paths)}")
    rows = [analyze(path) for path in paths]
    fields = [
        "image_id", "status", "mask_pixels", "green_reference",
        "green_regions", "green_pixels", "green_mask_pct",
        "hybrid_processed_pixels", "hybrid_regions", "hybrid_pixels",
        "hybrid_mask_pct", "hybrid_q0",
    ]
    with args.output.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
