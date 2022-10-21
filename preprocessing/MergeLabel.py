import glob
import os
# import gzip
# import shutil
# import random
import errno
# # import pydicom
import nibabel as nib


def merge_label_binary(path):
    all_niis = sorted(glob.glob(os.path.join(path, '*.nii.gz*')))
    print(len(all_niis))
    for nii_path in all_niis:
        nii = nib.load(nii_path)
        niidata = nii.get_fdata()
        niidata[niidata > 0] = 1

        segmentation_nii = nib.Nifti1Image(niidata,
                                           nii.affine,
                                           nii.header)
        nib.save(segmentation_nii, nii_path)

    print('done')


if __name__ == '__main__':
    merge_label_binary('/home/moucheng/projects_data/Task08_HepaticVessel/ssl/validate/lbls')
