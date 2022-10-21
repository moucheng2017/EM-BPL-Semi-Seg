import os
import sys

import matplotlib.pyplot as plt
import torch
import numpy as np

import torch
import random

import numpy as np
import scipy.ndimage

sys.path.append('..')
from libs.Augmentations import *

if __name__ == "__main__":
    # choose image volume:
    img_no = 0
    # choose which slice:
    slice_no = 1
    # read the file
    img_folder = '/home/moucheng/projects_data/Pulmonary_data/carve/3d_binary/R176/C3/D3_S3/N2/labelled/patches'
    lbl_folder = '/home/moucheng/projects_data/Pulmonary_data/carve/3d_binary/R176/C3/D3_S3/N2/labelled/labels'

    all_imgs = os.listdir(img_folder)
    all_imgs = [os.path.join(img_folder, img) for img in all_imgs]
    all_imgs.sort()

    all_lbls = os.listdir(lbl_folder)
    all_lbls = [os.path.join(lbl_folder, lbl) for lbl in all_lbls]
    all_lbls.sort()

    img = all_imgs[img_no]
    img = np.load(img)
    # img = img[1, :, :]

    lbl = all_lbls[img_no]
    lbl = np.load(lbl)
    # lbl = lbl[1, :, :]

    # # random zoom in augmentation:
    # zoom = RandomZoom()
    # img_zoomed, lbl_zoomed = zoom.forward(img, lbl)
    #
    # # plot
    # fig, axs = plt.subplots(nrows=2, ncols=2)
    # fig.suptitle('Random zoom in augmentation')
    # fig.set_figheight(10)
    # fig.set_figwidth(10)
    #
    # axs[0, 0].imshow(img[slice_no, :, :])
    # axs[0, 0].set_title('original image')
    #
    # axs[0, 1].imshow(img_zoomed[slice_no, :, :])
    # axs[0, 1].set_title('zoomed in image')
    #
    # axs[1, 0].imshow(lbl[slice_no, :, :])
    # axs[1, 0].set_title('original label')
    #
    # axs[1, 1].imshow(lbl_zoomed[slice_no, :, :])
    # axs[1, 1].set_title('zoomed in label')
    #
    # plt.show()
    #
    # # random cut out segmentation:
    # img_tensor = torch.from_numpy(img).cuda()
    # img_tensor = img_tensor.unsqueeze(0)
    #
    # lbl_tensor = torch.from_numpy(lbl).cuda()
    # lbl_tensor = lbl_tensor.unsqueeze(0)
    #
    # img_cut, lbl_cut = randomcutout(img_tensor, lbl_tensor)
    # img_cut, lbl_cut = img_cut.squeeze().cpu().numpy(), lbl_cut.squeeze().cpu().numpy()
    #
    # fig2, axs = plt.subplots(nrows=2, ncols=2)
    # fig2.suptitle('Random cutout in augmentation')
    # fig2.set_figheight(10)
    # fig2.set_figwidth(10)
    #
    # axs[0, 0].imshow(img[slice_no, :, :])
    # axs[0, 0].set_title('original image')
    #
    # axs[0, 1].imshow(img_cut[slice_no, :, :])
    # axs[0, 1].set_title('random cutout in image')
    #
    # axs[1, 0].imshow(lbl[slice_no, :, :])
    # axs[1, 0].set_title('original label')
    #
    # axs[1, 1].imshow(lbl_cut[slice_no, :, :])
    # axs[1, 1].set_title('random cutout in label')
    #
    # plt.show()

    # random contrast:
    contrast = RandomContrast()
    img_contrast_random = contrast.randomintensity(img)

    # plot
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Random contrast in augmentation')
    fig.set_figheight(5)
    fig.set_figwidth(5)

    axs[0].imshow(img[slice_no, :, :])
    axs[0].set_title('original image')

    axs[1].imshow(img_contrast_random[slice_no, :, :])
    axs[1].set_title('random contrast')

    plt.show()

    # random gaussian noise:







