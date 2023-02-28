import os

import matplotlib.pyplot as plt
import numpy as np

from libs.Augmentations import *
from PIL import Image

if __name__ == '__main__':
    # check the image intensities:
    # img_path = '/home/moucheng/projects_data/GLE689_mismatch_data/GLE_689_substack_im/2.25um_GLE-698_pag-0.05_20000.tif'
    # img = Image.open(img_path)
    # img = np.asfarray(img)
    # img = img / 255.
    # plt.imshow(img)
    # plt.show()
    ## conclusions:
    # the background intensity is 135
    # the empty areas intensities are 0.0

    img_path = '/home/moucheng/projects_data/PPFE_HipCT/GLE_689_substack_im'
    lbl_path = '/home/moucheng/projects_data/PPFE_HipCT/Labels_tif'

    img_save_name = '/home/moucheng/projects_data/PPFE_HipCT/processed/img_volume.npy'
    lbl_save_name = '/home/moucheng/projects_data/PPFE_HipCT/processed/lbl_volume.npy'

    starting = 450
    edges = 200

    all_slices = os.listdir(img_path)
    all_labels = os.listdir(lbl_path)

    all_slices.sort()
    all_labels.sort()

    all_slices = [os.path.join(img_path, a) for a in all_slices]
    all_labels = [os.path.join(lbl_path, a) for a in all_labels]

    for i, values in enumerate(zip(all_slices, all_labels)):

        img = Image.open(values[0])
        img = np.asfarray(img)
        img = np.expand_dims(img, axis=0)

        lbl = Image.open(values[1])
        lbl = np.asfarray(lbl)
        lbl = np.expand_dims(lbl, axis=0)

        if i == 0:
            img_volume = img
            lbl_volume = lbl
        else:
            img_volume = np.concatenate((img_volume, img), axis=0)
            lbl_volume = np.concatenate((lbl_volume, lbl), axis=0)

        print('slice ' + str(i) + ' done')

    img_volume = img_volume[starting:, edges:-edges, edges:-edges]
    lbl_volume = lbl_volume[starting:, edges:-edges, edges:-edges]
    # u, indices = np.unique(lbl_volume)

    img_volume = img_volume / 255.
    np.save(img_save_name, img_volume)
    np.save(lbl_save_name, lbl_volume)

    print(np.unique(lbl_volume))

    print('slices saved from the tiff images.')

    # # np.save('')
    # slice_index = 100
    # plt.imshow(img_volume[slice_index, :, :])
    # plt.show()


