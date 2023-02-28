import glob
import os
# import gzip
import shutil
# import random
import errno
import numpy as np


def read_all_files(path):
    all_files = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            all_files.append(os.path.join(path, name))
            # print(os.path.join(path, name))
    return all_files


# def separate_files(all_files_path, save_root_path):
#     path_for_each_resolution = {}


if __name__ == '__main__':

    allfiles = read_all_files('/home/moucheng/projects_data/Pulmonary_data/airway')

    save512img = '/home/moucheng/projects_data/Pulmonary_data/airway/512/imgs'
    save512lbl = '/home/moucheng/projects_data/Pulmonary_data/airway/512/lbls'

    save768img = '/home/moucheng/projects_data/Pulmonary_data/airway/768/imgs'
    save768lbl = '/home/moucheng/projects_data/Pulmonary_data/airway/768/lbls'

    dim1 = 0
    dim2 = 0

    dim1_paths = []
    dim2_paths = []

    for file in allfiles:
        data = np.load(file)
        # print(np.shape(file))
        c, d, h, w = np.shape(data)
        if h == 512:
            dim1 += 1
            if 'volume' in file:
                dim1_paths.append(file)
                filename = os.path.split(file)[-1]
                savepath = os.path.join(save512img, filename)
                shutil.move(file, savepath)
            elif 'label' in file:
                filename = os.path.split(file)[-1]
                savepath = os.path.join(save512lbl, filename)
                shutil.move(file, savepath)

        elif h == 768:
            dim2 += 1
            if 'volume' in file:
                dim2_paths.append(file)
                filename = os.path.split(file)[-1]
                savepath = os.path.join(save768img, filename)
                shutil.move(file, savepath)
            elif 'label' in file:
                filename = os.path.split(file)[-1]
                savepath = os.path.join(save768lbl, filename)
                # print(savepath)
                shutil.move(file, savepath)

    print('cases of 512:' + str(dim1//2))
    for file in dim1_paths:
        print(file)

    print('\n')
    print('\n')

    print('case of 768:' + str(dim2//2))
    for file in dim2_paths:
        print(file)