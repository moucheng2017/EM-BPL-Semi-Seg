import glob
import os
# import gzip
# import shutil
# import random
import errno
# # import pydicom
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize
# from nipype.interfaces.ants import N4BiasFieldCorrection
from tifffile import imsave


def prepare_data(data_path, lbl_path, save_img_path, save_lbl_path, case_index, labelled=True):
    # make directory first:
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_lbl_path, exist_ok=True)

    # # clean up the directory:
    # files = os.listdir(save_img_path)
    # if len(files) > 0:
    #     for f in files:
    #         os.remove(os.path.join(save_img_path, f))
    #
    # files = os.listdir(save_lbl_path)
    # if len(files) > 0:
    #     for f in files:
    #         os.remove(os.path.join(save_lbl_path, f))

    #  normalising intensities:
    data = nib.load(data_path)
    data = data.get_fdata()

    # applying lung window:
    data[data < -1000.0] = -1000.0
    data[data > 500.0] = 500.0
    data = (data - np.nanmean(data)) / np.nanstd(data)
    lbl = nib.load(lbl_path)
    lbl = lbl.get_fdata()

    if labelled is False:
        lbl = 100*np.ones_like(lbl)
    else:
        pass

    new_data = data[:, :, :]
    new_data = np.transpose(new_data, (2, 0, 1))
    new_data = np.expand_dims(new_data, axis=0)

    new_lbl = lbl[:, :, :]
    new_lbl = np.transpose(new_lbl, (2, 0, 1))
    new_lbl = np.expand_dims(new_lbl, axis=0)

    img_slice_store_name = case_index + '_volume.npy'
    gt_slice_store_name = case_index + '_label.npy'
    img_slice_store_name = save_img_path + '/' + img_slice_store_name
    gt_slice_store_name = save_lbl_path + '/' + gt_slice_store_name
    np.save(img_slice_store_name, new_data)
    np.save(gt_slice_store_name, new_lbl)


if __name__ == '__main__':

    data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/Turkish/raw/imgs/'
    lbl_path = '/home/moucheng/projects_data/Pulmonary_data/airway/Turkish/raw/lbls/'

    save_img_path = '/home/moucheng/projects_data/Pulmonary_data/airway/Turkish/labelled/imgs/'
    save_lbl_path = '/home/moucheng/projects_data/Pulmonary_data/airway/Turkish/labelled/lbls/'

    all_volumes = [os.path.join(data_path, i) for i in os.listdir(data_path)]
    all_lbls = [os.path.join(lbl_path, i) for i in os.listdir(lbl_path)]

    assert len(all_volumes) == len(all_lbls)
    print(str(len(all_volumes)) + ' cases are here.\n')

    all_volumes.sort()
    all_lbls.sort()

    for (img, lbl) in zip(all_volumes, all_lbls):
        # print(img)
        # print(lbl)
        case_index = os.path.split(img)[1].split('.')[0]
        prepare_data(img, lbl, save_img_path, save_lbl_path, case_index, labelled=True)
        print(case_index + ' is done.\n')

print('End')





