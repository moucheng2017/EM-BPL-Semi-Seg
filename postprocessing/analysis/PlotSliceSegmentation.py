import sys
sys.path.append("../..")
# torch.manual_seed(0)
import errno
import numpy as np
# import pandas as pd
import os
from os import listdir
# import Image

import timeit

import glob

import random
import matplotlib.pyplot as plt
# from skimage import exposure

from PIL import Image
from torch.utils import data

import nibabel as nib


def plot_slice(segmentation_path, data_path, label_path, slice_index, format='numpy'):
    if format == 'numpy':
        seg_numpy = np.load(segmentation_path, allow_pickle=True)
        seg_numpy = np.squeeze(seg_numpy)
        # print(np.shape(seg_numpy))

        data_numpy = np.load(data_path, allow_pickle=True)
        data_numpy = np.squeeze(data_numpy)
        # print(np.shape(data_numpy))

        lbl_numpy = np.load(label_path, allow_pickle=True)
        lbl_numpy = np.squeeze(lbl_numpy)
        # print(np.shape(lbl_numpy))
    elif format == 'nii':
        seg_numpy = nib.load(segmentation_path)
        seg_numpy = seg_numpy.get_fdata()
        seg_numpy = np.squeeze(seg_numpy)
        print(np.shape(seg_numpy))

        data_numpy = np.load(data_path, allow_pickle=True)
        data_numpy = np.squeeze(data_numpy)
        # print(np.shape(data_numpy))

        lbl_numpy = np.load(label_path, allow_pickle=True)
        lbl_numpy = np.squeeze(lbl_numpy)
        # print(np.shape(lbl_numpy))

    seg_slice = seg_numpy[slice_index, :, :]
    data_slice = data_numpy[slice_index, :, :]
    lbl_slice = lbl_numpy[slice_index, :, :]

    f, axarr = plt.subplots(3, figsize=(50, 150))
    axarr[0].imshow(data_slice, cmap='gray')
    axarr[1].imshow(seg_slice, cmap='gray')
    axarr[2].imshow(lbl_slice, cmap='gray')

    plt.show()


if __name__ == '__main__':

    path_seg = '/home/moucheng/projects_codes/Results/airway/Turkish/20220202/sup_unet_e1_l0.0001_b2_w16_s4000_d3_z32_x256/segmentation/Pat20a_volume_seg.nii'
    path_data = '/home/moucheng/projects_data/Pulmonary_data/airway/Mixed/validate/imgs/Pat20a_volume.npy'
    path_lbl = '/home/moucheng/projects_data/Pulmonary_data/airway/Mixed/validate/lbls/Pat20a_label.npy'

    plot_slice(segmentation_path=path_seg,
               data_path=path_data,
               label_path=path_lbl,
               slice_index=250,
               format='nii')

