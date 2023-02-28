import sys
sys.path.append('../..')
import matplotlib.pyplot as plt
import numpy as np
# from libs import norm95
from libs.Augmentations import *

if __name__ == '__main__':

    direction = 0
    slice_index = 100
    class_target = 1
    contrast_bin_no = 200
    # slice_index = 300
    # seg_path = '/home/moucheng/Results_class1.0/Segmentation/4_seg_2Dorthogonal_thresh0.99_temp2.0_ctr100.npy'
    # image_path = '/home/moucheng/projects_data/COVID_ML_data/COVID-CNN/validation_dataset/stack_4.npy'
    # label_path = '/home/moucheng/projects_data/COVID_ML_data/COVID-CNN/validation_dataset/stack_4_labels.npy'

    seg_path = '/home/moucheng/Results_class1.0/PPFE/0_seg_2Dorthogonal_thresh0.99_temp2.0_ctr200.npy'
    image_path = '/home/moucheng/projects_data/PPFE_HipCT/processed/imgs/0img.npy'
    # label_path = '/home/moucheng/projects_data/COVID_ML_data/COVID-CNN/validation_dataset/stack_4_labels.npy'

    images = np.load(image_path)
    images = norm95(images)
    seg = np.load(seg_path)
    seg = norm95(seg)

    # print(np.shape(images))
    # print(np.shape(seg))

    # labels = np.load(label_path)
    # labels[labels != class_target] = 0

    if direction == 0:
        img = np.squeeze(images[slice_index, :, :])
        seg = np.squeeze(seg[slice_index, :, :])
        # lbl = np.squeeze(labels[slice_index, :, :])
    elif direction == 1:
        img = np.squeeze(images[:, slice_index, :])
        seg = np.squeeze(seg[:, slice_index, :])
        # lbl = np.squeeze(labels[:, slice_index, :])
    else:
        img = np.squeeze(images[:, :, slice_index])
        # lbl = np.squeeze(labels[:, :, slice_index])

    img = resize(img, (256, 256), order=1)
    seg = resize(seg, (256, 256), order=0)
    # lbl = resize(lbl, (256, 256), order=0)

    fig = plt.figure(figsize=(5, 10))
    ax = []

    ax.append(fig.add_subplot(1, 2, 1))
    ax[-1].set_title('Image')

    if direction == 0:
        ax[-1].set_title('Image on D plane')
    elif direction == 1:
        ax[-1].set_title('Image on H plane')
    elif direction == 2:
        ax[-1].set_title('Image on W plane')

    plt.imshow(img, cmap='gray')
    plt.axis('off')

    ax.append(fig.add_subplot(1, 2, 2))
    ax[-1].set_title('seg')
    plt.imshow(seg, cmap='gray')
    plt.axis('off')

    # ax.append(fig.add_subplot(1, 3, 3))
    # ax[-1].set_title('Label')
    # plt.imshow(lbl, cmap='gray')
    # plt.axis('off')
    # if direction == 0:
    #     plt.savefig(str(contrast_bin_no) + '_s' + str(slice_index) + '_image_d_seg_lbl.png', bbox_inches='tight')
    # elif direction == 1:
    #     plt.savefig(str(contrast_bin_no) + '_s' + str(slice_index) + '_image_h_seg_lbl.png', bbox_inches='tight')
    # else:
    #     plt.savefig(str(contrast_bin_no) + '_s' + str(slice_index) + '_image_w_seg_lbl.png', bbox_inches='tight')

    if direction == 0:
        plt.savefig(str(contrast_bin_no) + '_s' + str(slice_index) + '_image_d_seg_lbl_PPFE.png', bbox_inches='tight')
    elif direction == 1:
        plt.savefig(str(contrast_bin_no) + '_s' + str(slice_index) + '_image_h_seg_lbl_PPFE.png', bbox_inches='tight')
    else:
        plt.savefig(str(contrast_bin_no) + '_s' + str(slice_index) + '_image_w_seg_lbl_PPFE.png', bbox_inches='tight')

    plt.show()
