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

from lungmask import mask
import SimpleITK as sitk
from pathlib import Path
import argparse

import dicom2nifti

# def args_parser():
#     parser = argparse.ArgumentParser('', add_help=False)
#     parser.add_argument('inputsource', type=str, help='path to dir holding the dataset')
#     parser.add_argument('outputdir', default='lung', type=str, help='path to dir to save the resultant')
#     parser.add_argument('-l', '--IDs', help='delimited list input', type=str)
#     return parser

def main(source_path,
         output_path,
         IDs):

    for each_case in IDs:
        case_dicom_path = os.path.join(source_path, each_case)
        case_dicom_path = os.path.join(case_dicom_path, 'XXX')
        case_nifti_path = os.path.join(output_path, each_case)
        os.makedirs(case_nifti_path, exist_ok=True)
        dicom2nifti.convert_directory(case_dicom_path, case_nifti_path, compression=True, reorient=True)
        print("Case " + each_case + " is Done.")
        print("\n")


if __name__ == '__main__':
    main(source_path='/home/moucheng/projects_data/Pulmonary_data/Leuven familial fibrosis/RETICULAR_ID',
         output_path='/home/moucheng/projects_data/Pulmonary_data/Leuven familial fibrosis/nifti',
         IDs=["ID005_112014",
              "ID005_022022",
              "ID006_042014",
              "ID006_102021",
              "ID022_012020",
              "ID033_042021",
              "ID039_062021",
              "ID042_032021",
              "ID042_032022",
              "ID046_072017",
              "ID046_072021",
              "ID053_032021",
              "ID053_012022",
              "ID090_062021",
              "ID091_082021",
              "ID11_092020",
              "ID120_102021",
              "ID148_022022",
              "ID21_112021"])