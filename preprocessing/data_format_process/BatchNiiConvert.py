# By Ashkan Pakzad ashkanpakzad.github.io
# Script that uses dicom2nifti to convert a folder of dicom cases into nifti. Dicoms should already be anonymised.
# Assumes that each case has a number of dicom series that are seperated into folders and the desired image to convert
# is the one with the most number of slices within its directory.
# 27th February 2021: v0.1.0 initiated

import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk


def args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('inputsource', type=str, help='path to dir holding the dicom dataset')
    parser.add_argument('--outputsource', '-o', default='nii', type=str, help='path to dir to output to nii files')

    return parser

def main(args):

    # give parent paths
    parentdicompth = Path(args.inputsource)
    parentoutputpth = Path(args.outputsource)
    parentoutputpth.mkdir(exist_ok=True)

    # get all cases to execute on
    dcmlist = [x.name for x in parentdicompth.iterdir()]
    outlist = [x.name for x in parentoutputpth.iterdir()]

    # identify cases that are not already converted
    convertlistname = list(set(dcmlist) - set(outlist))
    convertlist = [parentdicompth/Path(x) for x in convertlistname]

    for case in tqdm(convertlist):
        # identify dir with greatest number of dicoms
        subdir = [x[0] for x in os.walk(case)]
        Nfiles = [None]*len(subdir)
        for ii in range(len(subdir)):
            Nfiles[ii] = len(os.listdir(subdir[ii]))
        dcmdir = subdir[np.argmax(Nfiles)]

        # convert to nifti without change to orientation
        outniipath = parentoutputpth/Path(case.name+'.nii.gz')
        try:
            convertseries(dcmdir, str(outniipath))
        except RuntimeError:
            print(f'FAILED :{case}')



def convertseries(inpath, outpath):
    # based on https://stackoverflow.com/a/71074428
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(inpath)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Added a call to PermuteAxes to change the axes of the data
    # image = sitk.PermuteAxes(image, [2, 1, 0])

    sitk.WriteImage(image, outpath)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('output dicom dataset as nii.gz', parents=[args_parser()])
    args = parser.parse_args()
    main(args)
    print('COMPLETE')
