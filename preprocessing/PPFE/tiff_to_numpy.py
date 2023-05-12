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


parser = argparse.ArgumentParser('Run inference on PPFE')
parser.add_argument('--img_source', type=str, help='source file', default='/home/moucheng/projects_data/PPFE_HipCT/processed/img_volume.npy')
parser.add_argument('--model_source', type=str, help='model path', default='/home/moucheng/PhD/Clinical/ppfe/20230206/Results_res256_type3/3d_ppfe_sup_binary_res256_new_longer/Unet3D_l_0.0003_b1_w8_i12400_l2_0.0005_c_True_t1.0/trained_models/Unet3D_l_0.0003_b1_w8_i12400_l2_0.0005_c_True_t1.0_ema.pt')
parser.add_argument('--save_path', type=str, help='save path', default='/home/moucheng/PhD/Clinical/ppfe/20230206/inference')
parser.add_argument('--new_dim', type=int, help='new dimension', default=192)
parser.add_argument('--confidence', type=float, help='new dimension', default=0.5)
parser.add_argument('--flag', type=str, help='model flag', default='ema')
args = parser.parse_args()


if __name__ == '__main__':

    img_path = '/home/moucheng/projects_data/PPFE_HipCT/GLE_689_substack_im'
    lbl_path = '/home/moucheng/projects_data/PPFE_HipCT/Labels_tif'
