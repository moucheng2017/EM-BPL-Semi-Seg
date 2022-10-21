import nibabel
import torch
import numpy as np
import os

import numpy.ma as ma

from libs.old.Dataloader import RandomContrast


# Work flow for each case:
# 1. Read nii.gz
# 1.5. Prepare an empty volume for storing slices
# 2. Transform nii.gz into np
# 3. Slice np file
# 4. Expand the dim of the slice for inference
# 5. Segmentation
# 6. Store them in NIFTI file format


def nii2np(file_path):
    # Read image:
    data = nibabel.load(file_path)
    data = data.get_fdata()
    data = np.array(data, dtype='float32')
    # Now applying lung window:
    data[data < -1000.0] = -1000.0
    data[data > 500.0] = 500.0
    # H X W X D --> D X H X W:
    data = np.transpose(data, (2, 0, 1))
    if len(np.shape(data)) == 3:
        data = np.expand_dims(data, axis=0) # 1 X D X H X W
    return data


def np2tensor(data):
    data = torch.from_numpy(data).to(device='cuda', dtype=torch.float32)
    if len(data.size()) == 2:
        data = torch.unsqueeze(data, dim=0)
    return data


def np2tensor_batch(data_list):
    # print(len(data_list))
    # new_data_list = deque()
    new_data_list = []
    for data in data_list:
        data = np2tensor(data)
        new_data_list.append(data)
    return new_data_list


def normalisation(lung, image):
    image_masked = ma.masked_where(lung > 0.5, image)
    lung_mean = np.nanmean(image_masked)
    lung_std = np.nanstd(image_masked)
    image = (image - lung_mean + 1e-10) / (lung_std + 1e-10)
    return image


def adjustcontrast(data,
                   lung_mask,
                   adjust_times=0):
    # outputs = deque()
    outputs = []
    if adjust_times == 0:
        data = normalisation(lung=lung_mask, image=data)
        return outputs.append(data)
    else:
        contrast_augmentation = RandomContrast(bin_range=[10, 250])
        for i in range(adjust_times):
            data_ = contrast_augmentation.randomintensity(data)
            data_ = normalisation(lung=lung_mask, image=data_)
            outputs.append(data_)
        return outputs


def segment_single_case(volume,
                        model,
                        new_size):

    c, d, h, w = volume.size()
    print('volume has ' + str(d) + ' slices')

    location_list = {
        "left_up": [0, 0],
        "center_up": [0, (w-new_size) // 2],
        "right_up": [0, w-new_size],
        "left_middle": [(h - new_size)//2, 0],
        "center_middle": [(h - new_size)//2, (w-new_size) // 2],
        "right_middle": [(h - new_size)//2, w-new_size],
        "left_bottom": [h-new_size, 0],
        "center_bottom": [h-new_size, (w-new_size) // 2],
        "right_bottom": [h-new_size, w-new_size]
    }

    # location_list = {
    #     "left_up": [0, 0],
    #     "right_up": [0, w-new_size],
    #     "center_middle": [(h - new_size)//2, (w-new_size) // 2],
    #     "left_bottom": [h-new_size, 0],
    #     "right_bottom": [h-new_size, w-new_size]
    # }

    seg = np.zeros_like(volume.cpu().detach().numpy().squeeze())
    seg = np.transpose(seg, (1, 2, 0))

    # print(seg.size())

    for dd in range(d):
        img = volume[:, dd, :, :]
        # use 9 samples:
        for each_location, each_coordiate in location_list.items():
            cropped = img[:, each_coordiate[0]:each_coordiate[0] + new_size, each_coordiate[1]:each_coordiate[1] + new_size]
            cropped = torch.unsqueeze(cropped, dim=0)
            seg_patch, _ = model(cropped)
            seg_patch = torch.sigmoid(seg_patch)
            seg[each_coordiate[0]:each_coordiate[0] + new_size, each_coordiate[1]:each_coordiate[1] + new_size, dd] = seg_patch.detach().cpu().numpy()
        # use sliding windows:
        for h_ in range(0, h-new_size, 40):
            for w_ in range(0, w-new_size, 40):
                cropped = img[:, h_:h_ + new_size, w_:w_ + new_size]
                cropped = torch.unsqueeze(cropped, dim=0)
                seg_patch, _ = model(cropped)
                seg_patch = torch.sigmoid(seg_patch)
                seg[h_:h_ + new_size, w_:w_ + new_size, dd] = seg_patch.detach().cpu().numpy()
        print('slice ' + str(dd) + ' is done...')

    return seg


def segment2D(test_data_path,
              lung_path,
              model_path,
              contrast_no,
              new_size):

    # nii --> np:
    data = nii2np(test_data_path)
    lung = nii2np(lung_path)

    # load model:
    model = torch.load(model_path)
    model.eval()

    if contrast_no > 1:
        # ensemble on random contrast augmented of image:
        augmented_data_list = adjustcontrast(data, lung, adjust_times=contrast_no)

        # np --> tensor:
        augmented_data_list = np2tensor_batch(augmented_data_list)

        # prepare output:
        output = np.zeros_like(augmented_data_list[0].cpu().detach().numpy().squeeze())
        output = np.transpose(output, (1, 2, 0))

        # inference on each contrast:
        for augmented_data in augmented_data_list:
            output += segment_single_case(augmented_data, model, new_size)

        # ensemble on all of the contrast
        output = output / len(augmented_data_list)

    else:
        data = np2tensor(data)
        output = segment_single_case(data, model, new_size)

    lung = np.transpose(lung.squeeze(), (1, 2, 0))
    output = output*lung
    output = np.where(output > 0.5, 1, 0)

    return np.squeeze(output)


def save_seg(save_path,
             save_name,
             nii_path,
             saved_data):

    nii = nibabel.load(nii_path)
    segmentation_nii = nibabel.Nifti1Image(saved_data,
                                           nii.affine,
                                           nii.header)
    save_path_nii = os.path.join(save_path, save_name)
    nibabel.save(segmentation_nii, save_path_nii)


if __name__ == "__main__":
    case = '6357B'
    new_size = 480
    contrast_aug = 0

    save_path = '/home/moucheng/projects_data/Pulmonary_data/airway/segmentation'
    data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/test/imgs/' + case + '.nii.gz'
    lung_path = '/home/moucheng/projects_data/Pulmonary_data/airway/test/lung/' + case + '_lunglabel.nii.gz'
    model_path = '/home/moucheng/PhD/2022_12_Clinical/22020411/Sup2D_e1_l0.001_b5_w64_s1200_d4_r0.01_z1_x480/trained_models/Sup2D_e1_l0.001_b5_w64_s1200_d4_r0.01_z1_x480_1196.pt'
    save_name = case + '_seg2D_contrast_aug_' + str(contrast_aug) + '.nii.gz'

    segmentation = segment2D(data_path,
                             lung_path,
                             model_path,
                             contrast_aug,
                             new_size)

    save_seg(save_path, save_name, data_path, segmentation)
    print(np.shape(segmentation))
    print('End')



