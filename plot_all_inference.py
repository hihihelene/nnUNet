import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def strip_base(fn: str) -> str:
    bn = os.path.basename(fn)
    if bn.endswith('.nii.gz'):
        return bn[:-7]
    return os.path.splitext(bn)[0]


def get_slice(arr: np.ndarray) -> np.ndarray:
    a = np.squeeze(arr)
    if a.ndim == 3:
        return a[:, :, a.shape[2] // 2]
    if a.ndim == 2:
        return a
    raise ValueError(f'Unsupported array shape for slicing: {arr.shape}')


def main():
    base_dir = os.path.dirname(__file__)
    inf_dir = os.path.join(base_dir, 'nnUNet_inference')
    pred_dir = os.path.join(base_dir, 'nnUnet_inference_output')

    out_dir = os.path.join(pred_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(inf_dir, '*.nii*')))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.nii*')))

    images = {strip_base(p): p for p in image_files}
    preds = {strip_base(p): p for p in pred_files}

    keys = sorted(set(images.keys()) | set(preds.keys()))
    if not keys:
        print('No .nii/.nii.gz files found in the inference folders.')
        return

    saved = 0
    for k in keys:
        if k not in images:
            print(f'Skipping {k}: no image file in nnUNet_inference')
            continue

        img_path = images[k]
        pred_path = preds.get(k)

        try:
            img = nib.load(img_path).get_fdata()
        except Exception as e:
            print(f'Failed loading image {img_path}: {e}')
            continue

        pred = None
        if pred_path:
            try:
                pred = nib.load(pred_path).get_fdata()
            except Exception as e:
                print(f'Failed loading prediction {pred_path}: {e}')
                pred = None

        try:
            img_slice = get_slice(img)
        except ValueError as e:
            print(f'Skipping {k}: {e}')
            continue

        pred_slice = None
        if pred is not None:
            try:
                pred_slice = get_slice(pred)
            except ValueError:
                pred_slice = None

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        im0 = axes[0].imshow(img_slice, cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')
        cbar = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im0.set_clim(0, 500)

        axes[1].imshow(img_slice, cmap='gray')
        if pred_slice is not None:
            axes[1].imshow(pred_slice, alpha=0.4)
        axes[1].set_title('Prediction Overlay')
        axes[1].axis('off')

        fig.suptitle(k)
        out_path = os.path.join(out_dir, f'{k}.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f'Saved {out_path}')
        saved += 1

    print(f'Done. Saved {saved} plot(s) to: {out_dir}')


if __name__ == '__main__':
    main()
