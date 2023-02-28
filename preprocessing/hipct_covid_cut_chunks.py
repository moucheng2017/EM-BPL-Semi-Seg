import os
import numpy as np
import pathlib
import sys
sys.path.append('..')
from libs.Augmentations import norm95


if __name__ == '__main__':

    #new binary class
    new_class = 3.0
    tag = 'labelled' # labelled

    # cut chunks along d dimension into cubes every 162 pixels
    img_path = '/home/moucheng/projects_data/COVID_ML_data/original_' + tag + '/imgs'
    lbl_path = '/home/moucheng/projects_data/COVID_ML_data/original_' + tag + '/lbls'

    save_path_img = '/home/moucheng/projects_data/hipct_covid/class' + str(new_class) + '/' + tag + '/imgs/'
    save_path_lbl = '/home/moucheng/projects_data/hipct_covid/class' + str(new_class) + '/' + tag + '/lbls/'

    pathlib.Path(save_path_img).mkdir(parents=True, exist_ok=True)
    pathlib.Path(save_path_lbl).mkdir(parents=True, exist_ok=True)

    imgs = os.listdir(img_path)
    lbls = os.listdir(lbl_path)

    imgs.sort()
    lbls.sort()

    imgs = [os.path.join(img_path, i) for i in imgs]
    lbls = [os.path.join(lbl_path, i) for i in lbls]

    count = 0
    for img, lbl in zip(imgs, lbls):

        print(img)
        print(lbl)
        print('\n')

        img = np.load(img)
        lbl = np.load(lbl)

        img = np.asfarray(img)
        img = norm95(img)
        lbl = np.asfarray(lbl)

        lbl[lbl == new_class] = 1.0
        lbl[lbl != new_class] = 0.0

        d, h, w = np.shape(img)
        sub_imgs = np.split(img, d // 160, axis=0)
        sub_lbls = np.split(lbl, d // 160, axis=0)
        for each_sub_img, each_sub_lbl in zip(sub_imgs, sub_lbls):
            np.save(save_path_img+str(count)+'img.npy', each_sub_img)
            np.save(save_path_lbl+str(count)+'lbl.npy', each_sub_lbl)
            count += 1

    print('Done')

