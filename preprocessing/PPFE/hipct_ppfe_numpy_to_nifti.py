import os
import nibabel as nib

import matplotlib.pyplot as plt
import numpy as np

from libs.Augmentations import *
from PIL import Image

if __name__ == '__main__':

    img_save_name = '/home/moucheng/projects_data/PPFE_HipCT/processed/img_volume.npy'
    lbl_save_name = '/home/moucheng/projects_data/PPFE_HipCT/processed/lbl_volume.npy'

    img = np.load(img_save_name)
    lbl = np.load(lbl_save_name)

    new_img = nib.Nifti1Image(img, affine=np.eye(4))
    new_lbl = nib.Nifti1Image(lbl, affine=np.eye(4))

    nib.save(new_img, '/home/moucheng/projects_data/PPFE_HipCT/processed/dicom_volume_image.nii')
    nib.save(new_lbl, '/home/moucheng/projects_data/PPFE_HipCT/processed/dicom_volume_label.nii')

    print('nii file saved.')

    # # np.save('')
    # slice_index = 100
    # plt.imshow(img_volume[slice_index, :, :])
    # plt.show()


