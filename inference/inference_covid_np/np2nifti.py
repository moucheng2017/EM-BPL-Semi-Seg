import nibabel as nib
from pathlib import Path
import sys
sys.path.append("..")
sys.path.append("../..")
import numpy as np
import os


def save_nifti(np_array, path, name):
    new_image = nib.Nifti1Image(np_array, affine=np.eye(4))
    nib.save(new_image, os.path.join(path, name))
    print(name + ' done')


def save_all_imgs(folder):
    all_images = os.listdir(folder)
    all_images = [os.path.join(folder, i) for i in all_images]
    for each_img in all_images:
        if 'np' in each_img:
            data = np.load(each_img)
            img_name = Path(each_img).stem
            save_name = img_name + '.nii.gz'
            save_nifti(data, folder, save_name)


if __name__ == "__main__":
    # folder = '/home/moucheng/projects_data/COVID_ML_data/COVID-CNN/validation_dataset'
    # folder = '/home/moucheng/Results_class1.0/Segmentation'
    folder = '/home/moucheng/Results_class1.0/PPFE'
    save_all_imgs(folder)
