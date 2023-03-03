import os
import csv
import numpy as np
import pathlib
import scipy
import math
import sys
import argparse
import nibabel as nib
import torch
from pathlib import Path
sys.path.append('..')
import matplotlib.pyplot as plt
from skimage.transform import resize
import skimage.morphology as morph
from skimage.measure import label


# This prints out a grid: rows are different images, each row has three columns of image, label, and overlapped image
parser = argparse.ArgumentParser('Run connectivity post processing on PPFE')
parser.add_argument('--source', type=str, help='source file', default='/home/moucheng/PhD/Clinical/ppfe/20230206/inference/ema_d192_c0.5_segmentation.nii')
parser.add_argument('--save_path', type=str, help='save path', default='/home/moucheng/PhD/Clinical/ppfe/20230206/inference')
parser.add_argument('--min_size', type=int, help='minimum size', default=342368)
parser.add_argument('--connectivity', type=int, help='connectivity', default=1)
args = parser.parse_args()


if __name__ == '__main__':
    img = nib.load(args.source)
    img = img.get_fdata().astype(bool)
    img = morph.remove_small_objects(img, min_size=args.min_size, connectivity=args.connectivity)
    img = img.astype(int)
    d, h, w = np.shape(img)

    # img = img[(d-1280)//2:(d-1280)//2+1280,
    #           (h-1280)//2:(h-1280)//2+1280,
    #           (w-1280)//2:(w-1280)//2+1280]

    print(np.unique(img))
    new_img = nib.Nifti1Image(img, affine=np.eye(4))
    save_name = str(args.min_size) + '_' + str(args.connectivity) + '_processed_' + Path(args.source).stem + '.nii'
    save_name = os.path.join(args.save_path, save_name)
    nib.save(new_img, save_name)

    print('Done')

