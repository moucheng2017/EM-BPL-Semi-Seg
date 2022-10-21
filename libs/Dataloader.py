import collections
from libs.Augmentations import *


def normalisation(label, image):
    # Case-wise normalisation
    # Normalisation using values inside of the foreground mask

    if label is None:
        lung_mean = np.nanmean(image)
        lung_std = np.nanstd(image)
    else:
        image_masked = ma.masked_where(label > 0.5, image)
        lung_mean = np.nanmean(image_masked)
        lung_std = np.nanstd(image_masked)

    image = (image - lung_mean + 1e-10) / (lung_std + 1e-10)
    return image


class CT_Dataset_Orthogonal(Dataset):
    '''
    Each volume should be at: Dimension X Height X Width
    Sequentially random augment image with multiple steps
    '''
    def __init__(self,
                 images_folder,
                 labels_folder=None,
                 sampling_weight=5,
                 lung_window=True,
                 normalisation=True,
                 gaussian_aug=True,
                 zoom_aug=True,
                 contrast_aug=True):

        # flags
        # self.labelled_flag = labelled
        self.contrast_aug_flag = contrast_aug
        self.gaussian_aug_flag = gaussian_aug
        self.normalisation_flag = normalisation
        self.zoom_aug_flag = zoom_aug

        # data
        self.imgs_folder = images_folder
        self.lbls_folder = labels_folder

        self.lung_window_flag = lung_window

        if self.contrast_aug_flag is True:
            self.augmentation_contrast = RandomContrast(bin_range=(20, 255))

        if self.gaussian_aug_flag is True:
            self.gaussian_noise = RandomGaussian()

        self.augmentation_cropping = RandomSlicingOrthogonal(discarded_slices=1,
                                                             zoom=zoom_aug,
                                                             sampling_weighting_slope=sampling_weight)

    def __getitem__(self, index):
        # Lung masks:
        # all_lungs = sorted(glob.glob(os.path.join(self.lung_folder, '*.nii.gz*')))
        # lung = nib.load(all_lungs[index])
        # lung = lung.get_fdata()
        # lung = np.array(lung, dtype='float32')
        # lung = np.transpose(lung, (2, 0, 1))

        # Images:
        all_images = sorted(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))
        imagename = all_images[index]

        # load image and preprocessing:
        image = nib.load(imagename)
        image = image.get_fdata()
        image = np.array(image, dtype='float32')

        # Extract image name
        _, imagename = os.path.split(imagename)
        imagename, imagetxt = os.path.splitext(imagename)

        # transform dimension:
        image = np.transpose(image, (2, 0, 1)) # (H x W x D) --> (D x H x W)

        # Now applying lung window:
        if self.lung_window_flag is True:
            image[image < -1000.0] = -1000.0
            image[image > 500.0] = 500.0

        if self.lbls_folder:
            # Labels:
            all_labels = sorted(glob.glob(os.path.join(self.lbls_folder, '*.nii.gz*')))
            label = nib.load(all_labels[index])
            label = label.get_fdata()
            label = np.array(label, dtype='float32')
            label = np.transpose(label, (2, 0, 1))

            image_queue = collections.deque()

            # Apply normalisation at each case-wise:
            if self.normalisation_flag is True:
                image = normalisation(label, image)

            image_queue.append(image)

            # Random contrast:
            if self.contrast_aug_flag is True:
                image_another_contrast = self.augmentation_contrast.randomintensity(image)
                image_queue.append(image_another_contrast)

            # Random Gaussian:
            if self.gaussian_aug_flag is True:
                image_noise = self.gaussian_noise.gaussiannoise(image)
                image_queue.append(image_noise)

            # weights:
            dirichlet_alpha = collections.deque()
            for i in range(len(image_queue)):
                dirichlet_alpha.append(1)
            dirichlet_weights = np.random.dirichlet(tuple(dirichlet_alpha), 1)

            # make a new image:
            image_weighted = [weight*img for weight, img in zip(dirichlet_weights[0], image_queue)]
            image_weighted = sum(image_weighted)

            # Apply normalisation at each case-wise again:
            if self.normalisation_flag is True:
                image_weighted = normalisation(label, image_weighted)

            # get slices by weighted sampling on each axis with zoom in augmentation:
            inputs_dict = self.augmentation_cropping.crop(image_weighted, label)

            return inputs_dict, imagename

        else:
            image_queue = collections.deque()

            # Apply normalisation at each case-wise:
            if self.normalisation_flag is True:
                image = normalisation(None, image)
            image_queue.append(image)

            # Random contrast:
            if self.contrast_aug_flag is True:
                image_another_contrast = self.augmentation_contrast.randomintensity(image)
                image_queue.append(image_another_contrast)

            # Random Gaussian:
            if self.gaussian_aug_flag is True:
                image_noise = self.gaussian_noise.gaussiannoise(image)
                image_queue.append(image_noise)

            # weights:
            dirichlet_alpha = collections.deque()
            for i in range(len(image_queue)):
                dirichlet_alpha.append(1)
            dirichlet_weights = np.random.dirichlet(tuple(dirichlet_alpha), 1)

            # make a new image:
            image_weighted = [weight*img for weight, img in zip(dirichlet_weights[0], image_queue)]
            image_weighted = sum(image_weighted)

            # Apply normalisation at each case-wise again:
            if self.normalisation_flag is True:
                image_weighted = normalisation(None, image_weighted)

            inputs_dict = self.augmentation_cropping.crop(image_weighted)

            return inputs_dict, imagename

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(glob.glob(os.path.join(self.imgs_folder, '*.nii.gz*')))


def getData(data_directory,
            train_batchsize,
            sampling_weight,
            norm=True,
            zoom_aug=True,
            contrast_aug=True,
            lung_window=True,
            unlabelled=2):
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

    train_dataset_labelled = CT_Dataset_Orthogonal(images_folder=train_image_folder_labelled,
                                                   labels_folder=train_label_folder_labelled,
                                                   sampling_weight=sampling_weight,
                                                   normalisation=norm,
                                                   zoom_aug=zoom_aug,
                                                   contrast_aug=contrast_aug,
                                                   lung_window=lung_window)

    train_loader_labelled = data.DataLoader(dataset=train_dataset_labelled,
                                            batch_size=train_batchsize,
                                            shuffle=True,
                                            num_workers=0,
                                            drop_last=True)

    # Unlabelled images data set and data loader:
    if unlabelled > 0:
        train_image_folder_unlabelled = data_directory + '/unlabelled/imgs'

        train_dataset_unlabelled = CT_Dataset_Orthogonal(images_folder=train_image_folder_unlabelled,
                                                         sampling_weight=sampling_weight,
                                                         zoom_aug=False,
                                                         normalisation=norm,
                                                         contrast_aug=contrast_aug,
                                                         lung_window=lung_window)

        train_loader_unlabelled = data.DataLoader(dataset=train_dataset_unlabelled,
                                                  batch_size=train_batchsize*unlabelled,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  drop_last=True)

        return {'train_data_l': train_dataset_labelled,
                'train_loader_l': train_loader_labelled,
                'train_data_u': train_dataset_unlabelled,
                'train_loader_u': train_loader_unlabelled}

    else:
        return {'train_data_l': train_dataset_labelled,
                'train_loader_l': train_loader_labelled}


if __name__ == '__main__':
    dummy_input = np.random.rand(512, 512, 480)


