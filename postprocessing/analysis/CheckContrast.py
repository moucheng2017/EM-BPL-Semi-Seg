import glob
import os
import random
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def plot_different_contrast_ct(volume_path, contrasts, slice_no):

    volume_nii = nib.load(volume_path)
    volume = volume_nii.get_fdata()
    print(np.shape(volume))
    volume = (volume - np.nanmean(volume)) / np.nanstd(volume)

    fig, axes = plt.subplots(nrows=len(slice_no), ncols=len(contrasts)+1, figsize=(20, 10))
    for i in range(len(slice_no)):
        axes[i, 0].imshow(volume[:, :, slice_no[i]], cmap='gray')
        axes[i, 0].set_title("original image" + str(slice_no[i]))

    for i, bin in enumerate(contrasts):
        for j, no in enumerate(slice_no):
            h, w, d = np.shape(volume)
            image_histogram, bins = np.histogram(volume.flatten(), bin, density=True)
            cdf = image_histogram.cumsum()  # cumulative distribution function
            cdf = 255 * cdf / cdf[-1]  # normalize
            output = np.interp(volume.flatten(), bins[:-1], cdf)
            output = np.reshape(output, (h, w, d))
            # output = (output - np.nanmean(output)) / np.nanstd(output)

            axes[j, i+1].imshow(output[:, :, no], cmap='gray')
            axes[j, i+1].set_title('Contrast ' + str(bin))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_different_contrast_ct(volume_path='/home/moucheng/projects_data/Task08_HepaticVessel/ssl/test/imgs/hepaticvessel_001.nii.gz',
                               contrasts=[150, 180, 255],
                               slice_no=[10, 20])



