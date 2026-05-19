import numpy as np
import json
from pathlib import Path

import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

# path where labels may be found
LABELS_PATH = "unprocessed_data"

# path where corresponding dicoms may be found
DICOMS_PATH = "C:\\Lung_Project\\Measurements"

# path where the prepared data should be saved
PREPARED_DATA_PATH = "prepared_data"

# nnUNet dataset structure
DATASET_ID = "100"
DATASET_NAME = "LungSegmentation"
NNUNET_RAW_BASE = "nnUNet_raw"
DATASET_FOLDER = f"Dataset{DATASET_ID}_{DATASET_NAME}"

# Axis used to iterate over slices in the 3D label volume.
# Change this to 0, 1, or 2 depending on how your volumes are oriented.
SLICE_AXIS = 2


def create_nnunet_directories():
    """Create the required nnUNet directory structure."""
    base_path = Path(NNUNET_RAW_BASE) / DATASET_FOLDER
    images_dir = base_path / "imagesTr"
    labels_dir = base_path / "labelsTr"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    return base_path, images_dir, labels_dir


def get_permutation_for_moveaxis(source_axis, dest_axis):
    """
    Compute the axis permutation for SimpleITK.PermuteAxes() that moves an axis from source to dest.
    """
    perm = list(range(3))
    if source_axis != dest_axis:
        axis = perm.pop(source_axis)
        perm.insert(dest_axis, axis)
    return perm


def load_mitklabel_json(json_path):
    """Load and parse a .mitklabel.json file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_nifti_segmentation(nifti_path):
    """Load a NIfTI segmentation file and return the data array."""
    # Try different approaches to load the NIfTI file
    try:
        # Try nibabel first
        img = nib.load(nifti_path)
        return np.array(img.dataobj, dtype=np.int32)
    except Exception as e:
        print(f"Warning: Could not load {nifti_path} with nibabel: {e}")
        try:
            # Try SimpleITK as fallback
            img = sitk.ReadImage(nifti_path)
            return sitk.GetArrayFromImage(img).astype(np.int32)
        except Exception as e2:
            print(f"Error: Could not load {nifti_path} with SimpleITK either: {e2}")
            return None


def load_dicom_series(reference_files):
    """Load the matching DICOM series as a SimpleITK image and numpy array."""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in reference_files])
    image = reader.Execute()
    return sitk.GetArrayFromImage(image), image


def get_segmented_slices(segmentation_array):
    """
    Identify which slices contain segmentation (non-zero values).
    
    Args:
        segmentation_array: 3D array with slices along axis 0
    
    Returns:
        List of slice indices that contain segmentation
    """
    if segmentation_array is None:
        return []
    
    # Fast check: sum non-zero values along spatial axes (1, 2)
    per_slice_sum = np.count_nonzero(segmentation_array, axis=(1, 2))
    segmented_slices = np.where(per_slice_sum > 0)[0].tolist()
    
    return segmented_slices


def orient_slice_2d(slice_data):
    """Rotate label 90 degrees clockwise and mirror to match DICOM orientation."""
    return np.fliplr(np.rot90(slice_data, k=-1))


def extract_and_save_slices(nifti_path, reference_files, case_id, images_dir, labels_dir):
    """
    Extract segmented slices from a NIfTI file and save as individual images.

    Args:
        nifti_path: Path to the segmentation NIfTI file
        reference_files: Ordered DICOM files for the corresponding image volume
        case_id: Unique identifier for this case
        images_dir: Directory to save image slices
        labels_dir: Directory to save label slices

    Returns:
        Number of segmented slices extracted
    """
    # Load segmentation data
    seg_data = load_nifti_segmentation(nifti_path)
    if seg_data is None:
        return 0

    # Load the matching DICOM series so we can keep its orientation/spacing.
    image_data, image_volume = load_dicom_series(reference_files)

    # Ensure data is 3D before slicing.
    if seg_data.ndim != 3:
        print(f"Warning: {nifti_path} has unexpected dimensions: {seg_data.shape}")
        return 0

    if image_data.ndim != 3:
        print(f"Warning: DICOM series has unexpected dimensions: {image_data.shape}")
        return 0

    # Move the segmentation slice axis to position 0 for fast indexing.
    if SLICE_AXIS != 0:
        seg_data = np.moveaxis(seg_data, SLICE_AXIS, 0)
        image_volume = sitk.PermuteAxes(image_volume, get_permutation_for_moveaxis(SLICE_AXIS, 0))
    
    # Get segmented slices
    segmented_slices = get_segmented_slices(seg_data)

    if not segmented_slices:
        print(f"No segmented slices found in {case_id}")
        return 0

    print(
        f" Processing {case_id}: Found {len(segmented_slices)} segmented slices out of {seg_data.shape[0]}"
    )

    # Pre-compute 2D geometry metadata once to avoid expensive sitk.Extract() calls.
    # Extract a reference 2D image to get spacing and direction.
    # Note: image_volume may have been permuted, so we need its current size
    img_size = image_volume.GetSize()
    dicom_slice_template = sitk.Extract(image_volume, [img_size[0], img_size[1], 0], [0, 0, 0])

    # For each segmented slice, create a case (image and label pair)
    extracted_count = 0
    for slice_idx in segmented_slices:
        # Create a unique case ID for this slice
        slice_case_id = f"{case_id}_slice_{slice_idx:04d}"

        # Extract the slice (now fast since axis 0)
        dicom_slice = image_data[slice_idx]
        seg_slice = orient_slice_2d(seg_data[slice_idx])

        # Save image while preserving the DICOM spacing.
        image_filename = images_dir / f"{slice_case_id}_0000.nii.gz"
        image_output = sitk.GetImageFromArray(dicom_slice)
        # Set spacing and origin from the template, but use identity direction for 2D
        image_output.SetSpacing(dicom_slice_template.GetSpacing())
        image_output.SetOrigin(dicom_slice_template.GetOrigin())
        sitk.WriteImage(image_output, str(image_filename), True)

        # Save the label with the same geometry as the image slice.
        label_filename = labels_dir / f"{slice_case_id}.nii.gz"
        label_output = sitk.GetImageFromArray(seg_slice.astype(np.uint8))
        # Set spacing and origin from the template, but use identity direction for 2D
        label_output.SetSpacing(dicom_slice_template.GetSpacing())
        label_output.SetOrigin(dicom_slice_template.GetOrigin())
        sitk.WriteImage(label_output, str(label_filename), True)

        extracted_count += 1

    return extracted_count


def create_dataset_json(images_dir, labels_dir, num_training_cases):
    """
    Create the dataset.json file required by nnUNet.
    
    Args:
        images_dir: Path to imagesTr directory
        labels_dir: Path to labelsTr directory
        num_training_cases: Number of training cases
    """
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "lung": 1
        },
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
        "description": "Lung Segmentation Dataset - Individual slices extracted from 3D volumes"
    }
    
    json_path = images_dir.parent / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"Created dataset.json at {json_path}")


def main():
    """Main function to process all labeled data."""
    print("Starting data preparation for nnU-Net...")
    
    # Create directory structure
    base_path, images_dir, labels_dir = create_nnunet_directories()
    print(f"Created nnUNet directory structure at {base_path}")
    
    # Find all .mitklabel.json files
    labels_path = Path(LABELS_PATH)
    json_files = sorted(labels_path.glob("*.mitklabel.json"))
    
    if not json_files:
        print(f"No .mitklabel.json files found in {LABELS_PATH}")
        return
    
    print(f"Found {len(json_files)} label files")
    
    # Process each labeled volume
    total_slices_extracted = 0
    processed_cases = 0
    
    for json_file in tqdm(json_files, desc="Processing labeled volumes"):
        case_id = json_file.stem.replace(".mitklabel", "")
        
        try:
            # Load JSON metadata
            json_data = load_mitklabel_json(json_file)
            
            # Get the NIfTI file path
            if "groups" not in json_data or not json_data["groups"]:
                print(f"No groups found in {json_file}")
                continue
            
            nifti_filename = json_data["groups"][0]["_file"]
            nifti_path = labels_path / nifti_filename.lstrip("./")
            
            if not nifti_path.exists():
                print(f"Warning: NIfTI file not found: {nifti_path}")
                continue
            
            # Extract and save segmented slices
            reference_files = [Path(path) for _, path in json_data["properties"]["StringLookupTableProperty"]["referenceFiles"]]

            slices_extracted = extract_and_save_slices(
                nifti_path, reference_files, case_id, images_dir, labels_dir
            )
            
            if slices_extracted > 0:
                total_slices_extracted += slices_extracted
                processed_cases += 1
        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Create dataset.json
    if processed_cases > 0:
        create_dataset_json(images_dir, labels_dir, total_slices_extracted)
        print(f"\n[OK] Data preparation complete!")
        print(f"  - Processed {processed_cases} labeled volumes")
        print(f"  - Extracted {total_slices_extracted} segmented slices")
        print(f"  - Saved to {base_path}")
    else:
        print("No cases were successfully processed")


if __name__ == "__main__":
    main()

