import random
import nibabel as nib
import collections
import glob
import os
import numpy as np
from libs.Augmentations import RandomContrast, RandomCrop, RandomGaussian
from torch.utils import data
from torch.utils.data import Dataset


class CustomDataset3D(Dataset):
    def __init__(self,
                 images_folder,
                 labels_folder=None,
                 data_format='nii', # np for numpy and the default is nii
                 output_shape=(128, 128, 120),
                 mean_std=(0, 1),
                 transpose_dim=1,
                 crop_aug=1,
                 gaussian_aug=1,
                 contrast_aug=1
                 ):

        # flags
        self.contrast_aug_flag = contrast_aug
        self.gaussian_aug_flag = gaussian_aug
        self.crop_aug_flag = crop_aug
        self.data_format = data_format
        self.transpose_dim = transpose_dim

        # mean and std:
        self.mean_data = mean_std[0]
        self.std_data = mean_std[1]

        # data
        self.imgs_folder = images_folder
        self.lbls_folder = labels_folder

        if self.contrast_aug_flag == 1:
            self.augmentation_contrast = RandomContrast(bin_range=(20, 255))

        if self.gaussian_aug_flag == 1:
            self.gaussian_noise = RandomGaussian()

        if self.crop_aug_flag == 1:
            self.augmentation_crop = RandomCrop(output_shape)

    def __getitem__(self, index):
        # Check image extension:
        if self.data_format == 'np':
            all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.npy*')))
            imagename = all_images[index]
            image = np.load(imagename)
        elif self.data_format == 'nii':
            all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.nii.*')))
            imagename = all_images[index]
            image = nib.load(imagename)
            image = image.get_fdata()
        else:
            raise NotImplementedError

        image = np.array(image, dtype='float32')
        if self.transpose_dim > 0: # For images that are stored in (H x W x D), we transpose them to (D x H x W)
            if len(np.shape(image)) == 3: # Lung cancer
                image = np.transpose(image, (2, 0, 1))
            elif len(np.shape(image)) == 4: # BRATS
                # print(np.shape(image))
                image = np.transpose(image, (3, 2, 0, 1))
                # print(np.shape(image))
                # print(np.shape(image))
            else:
                raise NotImplementedError

        # Extract image name
        _, imagename = os.path.split(imagename)
        imagename, imagetxt = os.path.splitext(imagename)

        # Random Gaussian:
        if self.gaussian_aug_flag == 1:
            if random.random() > .5:
                image = self.gaussian_noise.gaussiannoise(image)

        # Random contrast:
        if self.contrast_aug_flag == 1:
            if random.random() > .5:
                image = self.augmentation_contrast.randomintensity(image)

        # Normalisation:
        image = (image - self.mean_data + 1e-10) / (self.std_data + 1e-10)

        if self.lbls_folder:
            # Labels:
            if self.data_format == 'np':
                all_labels = sorted(glob.glob(os.path.join(self.lbls_folder, '*.npy')))
                label = np.load(all_labels[index])
            elif self.data_format == 'nii':
                all_labels = sorted(glob.glob(os.path.join(self.lbls_folder, '*.nii.gz*')))
                label = nib.load(all_labels[index])
                label = label.get_fdata()
            else:
                raise NotImplementedError

            label = np.array(label, dtype='float32')
            label[label > 0] = 1 # make sure it is binary
            if self.transpose_dim > 0: # For images that are stored in (H x W x D), we transpose them to (D x H x W)
                label = np.transpose(label, (2, 0, 1))

            if self.crop_aug_flag == 1:
                image, label = self.augmentation_crop.crop_xy(image, label)
            return {'img': image, 'lbl': label}, imagename

        else:
            if self.crop_aug_flag == 1:
                image = self.augmentation_crop.crop_x(image)
            return {'img': image}, imagename

    def __len__(self):
        if self.data_format == 'np':
            return len(glob.glob(os.path.join(self.imgs_folder, '*.npy')))
        elif self.data_format == 'nii':
            return len(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz')))
        else:
            raise NotImplementedError


def getData3D(data_directory,
              transpose_dim,
              train_batchsize=1,
              data_format='np',
              contrast_aug=1,
              unlabelled=1,
              crop_aug=1,
              output_shape=(128, 128, 128),
              gaussian_aug=1,
              ):
    '''
    Args:
        data_directory:
        dataset_name:
        train_batchsize:
        norm:
        contrast_aug:
        lung_window:
        resolution:
        train_full:
        unlabelled:
    Returns:
    '''

    train_image_folder_labelled = data_directory + '/labelled/imgs'
    train_label_folder_labelled = data_directory + '/labelled/lbls'

    mean = 0.0
    var = 0.0

    if data_format == 'np':
        imgs = sorted(glob.glob(os.path.join(train_image_folder_labelled, '*.npy*')))
        lbls = sorted(glob.glob(os.path.join(train_label_folder_labelled, '*.npy*')))
        for image, label in zip(imgs, lbls):
            image = np.load(image)
            label = np.load(label)
            image = image[label > 0]
            mean += image.mean()
        mean /= len(imgs)

        for image, label in zip(imgs, lbls):
            image = np.load(image)
            label = np.load(label)
            image = image[label > 0]
            var += ((image - mean)**2).mean()
        var /= len(imgs)
        std = np.sqrt(var)

    elif data_format == 'nii':
        imgs = sorted(glob.glob(os.path.join(train_image_folder_labelled, '*.nii.gz*')))
        lbls = sorted(glob.glob(os.path.join(train_label_folder_labelled, '*.nii.gz*')))
        for image, label in zip(imgs, lbls):
            image = nib.load(image)
            image = image.get_fdata()
            label = nib.load(label)
            label = label.get_fdata()

            image = image[label > 0]
            mean += image.mean()
        mean /= len(imgs)

        for image, label in zip(imgs, lbls):
            image = nib.load(image)
            image = image.get_fdata()
            label = nib.load(label)
            label = label.get_fdata()
            image = image[label > 0]
            var += ((image - mean)**2).mean()
        var /= len(imgs)
        std = np.sqrt(var)

    else:
        raise NotImplementedError

    train_dataset_labelled = CustomDataset3D(images_folder=train_image_folder_labelled,
                                             labels_folder=train_label_folder_labelled,
                                             data_format=data_format,
                                             contrast_aug=contrast_aug,
                                             gaussian_aug=gaussian_aug,
                                             crop_aug=crop_aug,
                                             transpose_dim=transpose_dim,
                                             mean_std=(mean, std),
                                             output_shape=output_shape
                                             )

    train_loader_labelled = data.DataLoader(dataset=train_dataset_labelled,
                                            batch_size=train_batchsize,
                                            shuffle=True,
                                            drop_last=True)

    if unlabelled > 0:
        train_image_folder_unlabelled = data_directory + '/unlabelled/imgs'
        train_dataset_unlabelled = CustomDataset3D(images_folder=train_image_folder_unlabelled,
                                                   data_format=data_format,
                                                   contrast_aug=contrast_aug,
                                                   gaussian_aug=gaussian_aug,
                                                   crop_aug=crop_aug,
                                                   transpose_dim=transpose_dim,
                                                   mean_std=(mean, std),
                                                   output_shape=output_shape
                                                   )

        train_loader_unlabelled = data.DataLoader(dataset=train_dataset_unlabelled,
                                                  batch_size=int(train_batchsize*unlabelled),
                                                  shuffle=True,
                                                  drop_last=True)

        return {'train_data_l': train_dataset_labelled,
                'train_data_u': train_dataset_unlabelled,
                'train_loader_l': train_loader_labelled,
                'train_loader_u': train_loader_unlabelled}

    else:
        return {'train_data_l': train_dataset_labelled,
                'train_loader_l': train_loader_labelled}




