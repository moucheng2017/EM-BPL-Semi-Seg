import glob
import os
import errno
# # import pydicom
import numpy as np
import nibabel as nib


def numpy2nifti(source_np_file):
    source_np = np.load(source_np_file)
    new_img = nib.Nifti1Image(source_np, affine=np.eye(4))
    return new_img


def batch_numpy2nifti(source_np_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    all_numpys = os.listdir(source_np_folder)
    all_numpys = [os.path.join(source_np_folder, x) for x in all_numpys if '.npy' in x]
    for x in all_numpys:
        case_index = os.path.split(x)[1].split('.')[0]
        nifti_file = numpy2nifti(x)
        save_name = case_index + '.nii.gz'
        nib.save(nifti_file, os.path.join(save_folder, save_name))


if __name__ == '__main__':
    folder = '/home/moucheng/projects_data/HipCT/COVID_ML_data/hip_covid/inference/imgs'
    batch_numpy2nifti(folder, folder)
    print('COMPLETE')