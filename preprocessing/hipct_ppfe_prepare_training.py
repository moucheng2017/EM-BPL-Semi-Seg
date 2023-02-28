import os
import csv
import numpy as np
import pathlib
import scipy
import sys
import torch
sys.path.append('..')


if __name__ == '__main__':
    #  Change here:
    new_dim = 512
    label_merge = 3
    foreground_threshold_flag = False

    img_source = '/home/moucheng/projects_data/PPFE_HipCT/processed/img_volume.npy'
    lbl_source = '/home/moucheng/projects_data/PPFE_HipCT/processed/lbl_volume.npy'

    save_path_img = '/home/moucheng/projects_data/PPFE_HipCT/processed/res' + str(new_dim) + '_type' + str(label_merge) + '/labelled/imgs/'
    save_path_lbl = '/home/moucheng/projects_data/PPFE_HipCT/processed/res' + str(new_dim) + '_type' + str(label_merge) + '/labelled/lbls/'

    save_path_img_unlabelled = '/home/moucheng/projects_data/PPFE_HipCT/processed/res' + str(new_dim) + '_type' + str(label_merge) + '/unlabelled/imgs/'
    save_path_lbl_unlabelled = '/home/moucheng/projects_data/PPFE_HipCT/processed/res' + str(new_dim) + '_type' + str(label_merge) + '/unlabelled/lbls/'

    # ============================================================
    pathlib.Path(save_path_img).mkdir(parents=True, exist_ok=True)
    pathlib.Path(save_path_lbl).mkdir(parents=True, exist_ok=True)

    pathlib.Path(save_path_img_unlabelled).mkdir(parents=True, exist_ok=True)
    pathlib.Path(save_path_lbl_unlabelled).mkdir(parents=True, exist_ok=True)

    img_volume = np.load(img_source)
    lbl_volume = np.load(lbl_source)

    img = np.asfarray(img_volume)
    lbl = np.asfarray(lbl_volume)

    d, h, w = np.shape(img)
    print('The shape of the original volume. Depth: {:.1f}; ' 'Height: {:.1f}; ' 'Width: {:.1f}'.format(d, h, w))

    # stats of foreground areas:
    foreground_1 = np.zeros_like(lbl)
    foreground_2 = np.zeros_like(lbl)
    foreground_3 = np.zeros_like(lbl)
    foreground_1[lbl == 1] = 1
    foreground_2[lbl == 2] = 1
    foreground_3[lbl == 3] = 1
    foreground_1 = sum(sum(sum(foreground_1))) / (d*h*w)
    foreground_2 = sum(sum(sum(foreground_2))) / (d*h*w)
    foreground_3 = sum(sum(sum(foreground_3))) / (d*h*w)
    foreground_all = foreground_1 + foreground_2 + foreground_3
    print('The ratio of the foreground areas in percentages. class1: {:.10f}; ' 'class2: {:.10f}; ' 'class3: {:.10f}'.format(foreground_1*100, foreground_2*100, foreground_3*100))

    # with open('/home/moucheng/projects_data/PPFE_HipCT/processed/foreground_stats.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['class_index', 'area_percentage']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerow({'class_index': 'c1', 'area_percentage': str(foreground_1)})
    #     writer.writerow({'class_index': 'c2', 'area_percentage': str(foreground_2)})
    #     writer.writerow({'class_index': 'c3', 'area_percentage': str(foreground_3)})

    # Seems like 3 is nothing so let's merge the label 3 into class 2:
    lbl[lbl > 2] = 2
    if label_merge == 3:  # all foreground merged into one class
        lbl[lbl > 0] = 1
    elif label_merge == 2:  # only keep class 2 and binary
        lbl[lbl < 2] = 0
        lbl[lbl > 0] = 1
    elif label_merge == 1:  # only keep class 1 and binary
        lbl[lbl > 1] = 0

    # check there are only foregound and background labels:
    values = np.unique(lbl)
    print(values)
    assert values[0] == 0
    assert values[1] == 1
    assert len(values) == 2

    d_start = (d - (d // new_dim)*new_dim) // 2
    d_end = d_start + (d // new_dim)*new_dim

    h_start = (h - (h // new_dim)*new_dim) // 2
    h_end = h_start + (h // new_dim)*new_dim

    w_start = (w - (w // new_dim)*new_dim) // 2
    w_end = w_start + (w // new_dim)*new_dim

    img = img[d_start:d_end, h_start:h_end, w_start:w_end]
    lbl = lbl[d_start:d_end, h_start:h_end, w_start:w_end]
    d, h, w = np.shape(img)
    print('The shape of the trimmed volume. Depth: {:.1f}; ' 'Height: {:.1f}; ' 'Width: {:.1f}'.format(d, h, w))

    # normalise the whole image:
    img = (img - np.nanmean(img) + 1e-10) / (np.nanstd(img) + 1e-10)

    assert d % new_dim == 0
    assert h % new_dim == 0
    assert w % new_dim == 0

    imgs_d = np.split(img, d // new_dim, axis=0)
    lbls_d = np.split(lbl, d // new_dim, axis=0)
    count = 0

    for each_img_d, each_lbl_d in zip(imgs_d, lbls_d):
        imgs_d_h = np.split(each_img_d, h // new_dim, axis=1)
        lbls_d_h = np.split(each_lbl_d, h // new_dim, axis=1)
        for each_img_h, each_lbl_h in zip(imgs_d_h, lbls_d_h):
            imgs_d_h_w = np.split(each_img_h, w // new_dim, axis=2)
            lbls_d_h_w = np.split(each_lbl_h, w // new_dim, axis=2)
            for each_img_w, each_lbl_w in zip(imgs_d_h_w, lbls_d_h_w):
                d_, h_, w_ = np.shape(each_img_w)
                print('The shape of the saved subvolume. Depth: {:.1f}; ' 'Height: {:.1f}; ' 'Width: {:.1f}'.format(d_, h_, w_))
                assert d_ == new_dim
                assert h_ == new_dim
                assert w_ == new_dim

                # check the foreground:
                foreground = np.zeros_like(each_img_w)
                foreground[each_lbl_w == 1] = 1
                foreground = sum(sum(sum(foreground)))
                threshold = new_dim*new_dim*new_dim*foreground_all*0.5

                if foreground_threshold_flag is True:
                    if foreground > threshold:
                        np.save(save_path_img + str(count) + 'img.npy', each_img_w)
                        np.save(save_path_lbl + str(count) + 'lbl.npy', each_lbl_w)
                else:
                    np.save(save_path_img + str(count) + 'img.npy', each_img_w)
                    np.save(save_path_lbl + str(count) + 'lbl.npy', each_lbl_w)

                count += 1

    print('Done')

