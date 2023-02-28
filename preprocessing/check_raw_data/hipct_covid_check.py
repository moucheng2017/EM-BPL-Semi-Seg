import sys
sys.path.append('../..')
import matplotlib.pyplot as plt
import numpy as np
# from libs import norm95
from libs.Augmentations import *

if __name__ == '__main__':

    image_path = '/home/moucheng/projects_data/hipct_covid/class1.0/labelled/imgs/1img.npy'
    label_path = '/home/moucheng/projects_data/hipct_covid/class1.0/labelled/lbls/1lbl.npy'

    # image_path = '/home/moucheng/projects_data/hipct_covid/original_labelled/imgs/stack_0.npy'
    # label_path = '/home/moucheng/projects_data/hipct_covid/original_labelled/lbls/stack_0_labels.npy'

    images = np.load(image_path)
    images = norm95(images)
    labels = np.load(label_path)

    direction = 2
    slice_index = 50

    if direction == 0:
        img = np.squeeze(images[slice_index, :, :])
        lbl = np.squeeze(labels[slice_index, :, :])
    elif direction == 1:
        img = np.squeeze(images[:, slice_index, :])
        lbl = np.squeeze(labels[:, slice_index, :])
    else:
        img = np.squeeze(images[:, :, slice_index])
        lbl = np.squeeze(labels[:, :, slice_index])

    img = resize(img, (256, 256), order=1)
    lbl = resize(lbl, (256, 256), order=0)

    fig = plt.figure(figsize=(20, 30))
    ax = []

    ax.append(fig.add_subplot(1, 5, 1))
    ax[-1].set_title('Image')

    if direction == 0:
        ax[-1].set_title('Image on D plane')
    elif direction == 1:
        ax[-1].set_title('Image on H plane')
    elif direction == 2:
        ax[-1].set_title('Image on W plane')

    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # gaussian noise:
    img = np.expand_dims(img, axis=0)
    img = RandomGaussian().gaussiannoise(img)
    img = np.squeeze(img)
    # img = norm95(img)
    ax.append(fig.add_subplot(1, 5, 2))
    ax[-1].set_title('Gaussian Noise')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # contrast with histogram equalisation:
    img = np.expand_dims(img, axis=0)
    img = RandomContrast().randomintensity(img)
    img = np.squeeze(img)
    # img = norm95(img)
    ax.append(fig.add_subplot(1, 5, 3))
    ax[-1].set_title('Random contrast')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # random zoom:
    img = norm95(img)
    img, lbl = RandomZoom().forward(img, lbl)
    img = np.squeeze(img)
    # img = norm95(img)
    ax.append(fig.add_subplot(1, 5, 4))
    ax[-1].set_title('Random Zoom')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # label
    # print(np.unique(lbl))
    # lbl[lbl != 1] = 0
    ax.append(fig.add_subplot(1, 5, 5))
    ax[-1].set_title('Label')
    plt.imshow(lbl, cmap='gray')
    plt.axis('off')

    if direction == 0:
        plt.savefig('image_d.png', bbox_inches='tight')
    elif direction == 1:
        plt.savefig('image_h.png', bbox_inches='tight')
    else:
        plt.savefig('image_w.png', bbox_inches='tight')

    plt.show()
