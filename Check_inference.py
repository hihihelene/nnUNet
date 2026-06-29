import os

import nibabel as nib
import matplotlib.pyplot as plt

path_img = 'E:/LungSegmentation/nnUNet/nnUNet_inference'
path_label = 'E:/LungSegmentation/nnUNet/nnUnet_inference_output'

# load the file lists for both images and labels
img_files = sorted([f for f in os.listdir(path_img) if f.endswith('.nii.gz')])
label_files = sorted([f for f in os.listdir(path_label) if f.endswith('.nii.gz')])


#for every image look for the corresponding label and load them
for img_file in img_files:
    
        img_path = os.path.join(path_img, img_file)
        label_path = os.path.join(path_label, label_files[img_files.index(img_file)])

        img_nii = nib.load(img_path)
        label_nii = nib.load(label_path)

        img = img_nii.get_fdata()
        pred = label_nii.get_fdata()

        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.imshow(img[:,:], cmap="gray")
        plt.title("Image")
        plt.colorbar()
        plt.clim(0, 500)

        plt.subplot(1,2,2)
        plt.imshow(img[:,:], cmap="gray")
        plt.imshow(pred[:,:], alpha=0.4)
        plt.title("Prediction Overlay")


        plt.show()