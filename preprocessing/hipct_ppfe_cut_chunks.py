import os
import numpy as np
import pathlib
import scipy
import sys
sys.path.append('..')
from libs.Augmentations import norm95


if __name__ == '__main__':

    new_dim = 160

    # cut chunks along d dimension into cubes every 162 pixels
    img_path = '/home/moucheng/projects_data/PPFE_HipCT/processed/original_labelled'
    # lbl_path = '/home/moucheng/projects_data/COVID_ML_data/original_' + tag + '/lbls'

    save_path_img = '/home/moucheng/projects_data/PPFE_HipCT/processed/imgs/'
    # save_path_lbl = '/home/moucheng/projects_data/hipct_covid/class' + str(new_class) + '/' + tag + '/lbls/'

    pathlib.Path(save_path_img).mkdir(parents=True, exist_ok=True)
    # pathlib.Path(save_path_lbl).mkdir(parents=True, exist_ok=True)

    imgs = os.listdir(img_path)
    # lbls = os.listdir(lbl_path)

    imgs.sort()
    # lbls.sort()

    imgs = [os.path.join(img_path, i) for i in imgs]
    # lbls = [os.path.join(lbl_path, i) for i in lbls]

    count = 0

    for img in imgs:

        print(img)
        print('\n')

        img = np.load(img)
        img = np.asfarray(img)

        print(np.shape(img))
        d, h, w = np.shape(img)

        dd = new_dim*(d // new_dim)
        hh = h % new_dim // 2
        ww = w % new_dim // 2

        img = img[1412-800:, 311:311+800, 311:311+800]
        # img = img[d - dd:, hh:h - hh - 1, ww:w - ww - 1]
        d, h, w = np.shape(img)
        print(np.shape(img))

        sub_imgs = np.split(img, d // new_dim, axis=0)
        for each_sub_img in sub_imgs:
            sub_sub_imgs = np.split(each_sub_img, h // new_dim, axis=1)
            for each_sub_sub_img in sub_sub_imgs:
                sub_sub_sub_imgs = np.split(each_sub_sub_img, w // new_dim, axis=2)
                for each_sub_sub_sub_img in sub_sub_sub_imgs:
                    # each_sub_sub_sub_img = scipy.ndimage.zoom(input=each_sub_sub_sub_img, zoom=(1, 160 // new_dim, 160 // new_dim), order=0)
                    np.save(save_path_img + str(count) + 'img.npy', each_sub_sub_sub_img)
                    count += 1

    print('Done')

