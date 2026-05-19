from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parent
LABELS_DIR = ROOT / "unprocessed_data"
DEFAULT_OUTPUT_DIR = ROOT / "exemplary_plots"
DEFAULT_SLICE_AXIS = 2


def load_case_metadata(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_case_files(json_path: Path) -> tuple[Path, list[Path]]:
    metadata = load_case_metadata(json_path)

    if not metadata.get("groups"):
        raise RuntimeError(f"No groups found in {json_path}")

    seg_file = (json_path.parent / metadata["groups"][0]["_file"].lstrip("./")).resolve()
    if not seg_file.exists():
        raise FileNotFoundError(f"Segmentation file not found: {seg_file}")

    reference_files = metadata.get("properties", {}).get("StringLookupTableProperty", {}).get("referenceFiles", [])
    image_files = [Path(path).resolve() for _, path in reference_files]
    image_files = [path for path in image_files if path.exists()]

    if not image_files:
        raise FileNotFoundError(f"No reference DICOM files found for {json_path}")

    return seg_file, image_files


def load_dicom_volume(image_files: list[Path]) -> np.ndarray:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in image_files])
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)


def load_segmentation_volume(seg_file: Path) -> np.ndarray:
    try:
        image = nib.load(str(seg_file))
        return np.asanyarray(image.dataobj)
    except Exception:
        image = sitk.ReadImage(str(seg_file))
        return sitk.GetArrayFromImage(image)


def move_slice_axis_to_front(volume: np.ndarray, slice_axis: int) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    if slice_axis < 0 or slice_axis >= volume.ndim:
        raise ValueError(f"slice_axis must be in [0, {volume.ndim - 1}], got {slice_axis}")
    if slice_axis == 0:
        return volume
    return np.moveaxis(volume, slice_axis, 0)


def pick_slice_index(segmentation: np.ndarray) -> int:
    foreground = segmentation > 0
    per_slice = foreground.sum(axis=(1, 2))
    return int(np.argmax(per_slice))


def normalize_image(slice_2d: np.ndarray) -> np.ndarray:
    image = slice_2d.astype(np.float32)
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return image


def align_segmentation_to_dicom(seg_slice: np.ndarray) -> np.ndarray:
    """Rotate label 90 degrees clockwise and mirror to match DICOM orientation."""
    return np.fliplr(np.rot90(seg_slice, k=-1))


def plot_case(json_path: Path, output_path: Path, slice_axis: int) -> tuple[Path, int]:
    seg_file, image_files = resolve_case_files(json_path)
    image_volume = load_dicom_volume(image_files)
    segmentation_raw = load_segmentation_volume(seg_file)
    segmentation = move_slice_axis_to_front(segmentation_raw, slice_axis)

    if image_volume.shape != segmentation.shape:
        raise RuntimeError(
            f"Shape mismatch for {json_path.name}: image {image_volume.shape} vs seg {segmentation.shape}. "
            f"Check the slice axis setting."
        )

    selected_slice = pick_slice_index(segmentation)

    image_slice = normalize_image(image_volume[selected_slice])
    seg_slice = segmentation[selected_slice]
    seg_slice = align_segmentation_to_dicom(seg_slice)
    overlay = np.dstack([image_slice, image_slice, image_slice])
    mask = seg_slice > 0
    overlay[mask, 0] = 1.0
    overlay[mask, 1] = np.clip(overlay[mask, 1] * 0.35, 0.0, 1.0)
    overlay[mask, 2] = np.clip(overlay[mask, 2] * 0.35, 0.0, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    figure.suptitle(f"{json_path.stem} | slice {selected_slice} | axis {slice_axis}")

    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(seg_slice, cmap="gray", vmin=0)
    axes[1].set_title("Segmentation")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    return output_path, selected_slice


def find_cases(labels_dir: Path, limit: int | None) -> list[Path]:
    json_files = sorted(labels_dir.glob("*.mitklabel.json"))
    if limit is not None:
        json_files = json_files[:limit]
    return json_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot exemplary image/segmentation overlays for inspection.")
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR, help="Directory containing .mitklabel.json files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated figures")
    parser.add_argument("--slice-axis", type=int, default=DEFAULT_SLICE_AXIS, help="Axis used as the slice dimension in the segmentation volume")
    parser.add_argument("--count", type=int, default=3, help="Number of cases to plot")
    args = parser.parse_args()

    cases = find_cases(args.labels_dir, args.count)
    if not cases:
        raise RuntimeError(f"No .mitklabel.json files found in {args.labels_dir}")

    print(f"Plotting {len(cases)} case(s) with slice axis {args.slice_axis}")
    for json_path in cases:
        output_path = args.output_dir / f"{json_path.stem}_axis{args.slice_axis}.png"
        saved_path, selected_slice = plot_case(json_path, output_path, args.slice_axis)
        print(f"Saved {saved_path} (slice {selected_slice})")


if __name__ == "__main__":
    main()