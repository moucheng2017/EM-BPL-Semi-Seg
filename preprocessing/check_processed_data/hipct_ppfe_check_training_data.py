import os
import csv
import numpy as np
import pathlib
import scipy
import sys
import argparse
sys.path.append('../..')
import matplotlib.pyplot as plt
from skimage.transform import resize

# This prints out a grid: rows are different images, each row has three columns of image, label, and overlapped image

parser = argparse.ArgumentParser('Check training data')

parser.add_argument('--img_source', type=str, help='source image folder', default='/home/moucheng/projects_data/PPFE_HipCT/processed/res256_type3/labelled/imgs')
parser.add_argument('--lbl_source', type=str, help='source label folder', default='/home/moucheng/projects_data/PPFE_HipCT/processed/res256_type3/labelled/lbls')
parser.add_argument('--ids', type=str, default='200,250,255', help='sequence of a indeces of the slices, we suggest 5, if too many, you need to change the figure size')
parser.add_argument('--volume', type=int, default=40, help='volume index')
args = parser.parse_args()

if __name__ == '__main__':

    images_index = args.ids
    images_index = images_index.split(',')

    volume_index = args.volume
    img_volumes = os.listdir(args.img_source)
    lbl_volumes = os.listdir(args.lbl_source)
    img_volumes.sort()
    lbl_volumes.sort()
    all_imgs = [os.path.join(args.img_source, a) for a in img_volumes]
    all_lbls = [os.path.join(args.lbl_source, a) for a in lbl_volumes]

    img_volume = np.load(all_imgs[volume_index])
    lbl_volume = np.load(all_lbls[volume_index])

    img = np.asfarray(img_volume)
    lbl = np.asfarray(lbl_volume)

    d, h, w = np.shape(img)
    print('The shape of the original volume. Depth: {:.1f}; ' 'Height: {:.1f}; ' 'Width: {:.1f}'.format(d, h, w))
    print(sum(sum(sum(lbl))))

    rows = len(images_index)
    cols = 3
    f, axarr = plt.subplots(rows, cols, squeeze=False)

    for i in range(len(images_index)):

        slice_index = int(images_index[i])
        img_slice = img[slice_index, :, :]
        # img_slice = resize(np.squeeze(img_slice), (32, 32), order=1)

        lbl_slice = lbl[slice_index, :, :]
        print(sum(sum(lbl_slice)))
        # lbl_slice = resize(np.squeeze(lbl_slice), (32, 32), order=0)

        overlay = 0.3*img_slice + 0.7*lbl_slice

        images = [img_slice, lbl_slice, overlay]

        for j in range(3):
            axarr[i, j].imshow(images[j], cmap='gray')

    plt.show()

    # Could add here to save the image if needed

    print('Done')

