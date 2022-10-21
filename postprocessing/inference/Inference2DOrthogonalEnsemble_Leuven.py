import nibabel
import sys
import torch
sys.path.append("../..")
import numpy as np
import os
from pathlib import Path

import nibabel as nib

from tqdm import tqdm

import numpy.ma as ma

from libs.old.Dataloader import RandomContrast

from libs.old.Models2DOrthogonal import Unet2DMultiChannel


def nii2np(file_path):
    # Read image:
    data = nibabel.load(file_path)
    data = data.get_fdata()
    data = np.array(data, dtype='float32')
    # print(np.shape(data))
    # Now applying lung window:
    data[data < -1000.0] = -1000.0
    data[data > 500.0] = 500.0
    # H X W X D --> D X H X W:
    data = np.transpose(data, (2, 0, 1))
    d, h, w = np.shape(data)
    # print(np.shape(data))
    data = np.pad(data, pad_width=((0, 512 - d), (0, 512 - h), (0, 512 - w)), mode='symmetric')

    if len(np.shape(data)) == 3:
        data = np.expand_dims(data, axis=0) # 1 X D X H X W
    return data, d


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


def seg_one_plane(volume,
                   model,
                   direction=0,
                   temperature=2
                   ):

    c, d, h, w = volume.size()

    # seg = np.zeros_like(volume.cpu().detach().numpy().squeeze())
    # seg = np.transpose(seg, (1, 2, 0))
    seg = np.zeros((512, 512, 512))

    if direction == 0:
        slice_no = d
    elif direction == 1:
        slice_no = h
    elif direction == 2:
        slice_no = w

    for i in range(0, slice_no, 1):

        if direction == 0:
            img = volume[:, i, :, :] # B x 1 x 512 x 512
            img = img.unsqueeze(1)
        elif direction == 1:
            img = volume[:, :, i, :] # B x 512 x 1 x 512
            img = img.unsqueeze(1)
        elif direction == 2:
            img = volume[:, :, :, i] # B x 512 x 512 x 1
            img = img.unsqueeze(1)

        # print(img.size())
        seg_, _ = model(img)
        seg_ = torch.sigmoid(seg_ / temperature)
        seg_ = seg_.squeeze().detach().cpu().numpy() # W x D x H

        if direction == 0:
            seg[i, :, :] = seg_
        elif direction == 1:
            seg[:, i, :] = seg_
        elif direction == 2:
            seg[:, :, i] = seg_

    return seg


def seg_three_plaines(volume,
                      model,
                      lung,
                      temperature=2
                      ):

    # c, d, h, w = volume.size()
    # print(volume.size())

    seg0 = seg_one_plane(volume, model, 0, temperature)
    seg1 = seg_one_plane(volume, model, 1, temperature)
    seg2 = seg_one_plane(volume, model, 2, temperature)

    seg = (seg0 + seg1 + seg2) / 3

    del seg0
    del seg1
    del seg2

    # seg = seg.squeeze()[:d, :, :]
    # seg = np.transpose(seg, (1, 2, 0))

    # lung = np.transpose(lung.squeeze(), (1, 2, 0))

    seg = seg*lung
    del lung

    return seg


def save_seg(save_path,
             save_name,
             nii_path,
             saved_data):

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    nii = nibabel.load(nii_path)
    segmentation_nii = nibabel.Nifti1Image(saved_data,
                                           nii.affine,
                                           nii.header)
    save_path_nii = os.path.join(save_path, save_name)
    nibabel.save(segmentation_nii, save_path_nii)


def ensemble(seg_path,
             threshold):

    final_seg = []
    all_segs = os.listdir(seg_path)
    all_segs.sort()
    all_segs = [os.path.join(seg_path, seg_name) for seg_name in all_segs if 'prob' in seg_name]

    for seg_path in all_segs:
        seg = nib.load(seg_path)
        seg = seg.get_fdata()
        seg = np.array(seg, dtype='float32')
        final_seg.append(seg)
        os.remove(seg_path)

    output = sum(final_seg) / len(final_seg)
    output = np.where(output > threshold, 1, 0)
    del final_seg
    return output


def main(test_data_path,
         lung_path,
         model_path,
         temp,
         dataset,
         threshold=0.8,
         step_lower=20000,
         step_upper=20200):

    # generate save path:
    # /data/model/config/nii.gz

    path = Path(os.path.abspath(model_path))
    save_path_parent = path.parent.absolute()
    save_path = os.path.join(save_path_parent, 'segmentation_thresh' + str(threshold) + '_temp' + str(temp))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataname = Path(test_data_path).name
    save_path = os.path.join(save_path, dataname)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # sort out all models:
    all_models = os.listdir(model_path)
    all_models.sort()

    # ran inference for each model:
    for model_name in tqdm(all_models):

        step = model_name.split('_')[-1]
        step = float(step.split('.')[0])

        if step_lower < step < step_upper:
            model_name = os.path.join(model_path, model_name)

            # nii --> np:
            data, d1 = nii2np(test_data_path)
            lung, d2 = nii2np(lung_path)
            assert d1 == d2

            model = Unet2DMultiChannel(in_ch=1, width=24, output_channels=1)
            model.to('cuda')
            checkpoint = torch.load(model_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # np --> tensor
            data = np2tensor(data)

            # segmentation 3 orthogonal planes:
            seg = seg_three_plaines(data, model, lung, temp)

            # seg = np.transpose(np.squeeze(seg), (2, 0, 1))
            seg = np.squeeze(seg)
            seg = seg[:d1, :, :]
            seg = np.transpose(seg, (1, 2, 0))

            # save prepration:
            save_name = str(step) + '_prob.nii.gz'

            # save seg:
            save_seg(save_path,
                     save_name,
                     test_data_path,
                     seg)

    # ensemble all segmentation files:
    final_seg = ensemble(save_path, threshold)
    save_name = 'final_seg.nii.gz'

    # save seg:
    save_seg(save_path,
             save_name,
             test_data_path,
             final_seg)

    print(dataname + ' is done.')


if __name__ == "__main__":

    # Hyperparameters:
    dataset_tag = 'Leuven'
    cases = ['ID005_022022/7_inspiratie_rug_long_10__br58_vol',
             'ID005_022022/9_expiratie_rug_long_10__br59_vol',
             'ID039_062021/8_inspiratie_body_10__inspiratie',
             'ID039_062021/12_expiratie_lcad_lung_10__expiratie_lcad',
             'ID039_062021/14_expiratie_body_10__expiratie',
             'ID091_082021/4_expiratie_rug_long_10__b70f_vol',
             'ID091_082021/8_inspiratie_rug_long_10__b60f_vol'] # testing case

    threshold = 0.4 # confidence threshold
    temperature = 2 # temperature scaling for sigmoid/softmax

    for case in cases:
        data_path = '/home/moucheng/projects_data/Pulmonary_data/Leuven familial fibrosis/nifti/' + case + '.nii.gz'
        lung_path = '/home/moucheng/projects_data/Pulmonary_data/Leuven familial fibrosis/lungmask/' + case + '_lunglabel.nii.gz'
        model_path = '/home/moucheng/projects_codes/Results/airway/2022_07_04/OrthogonalSup2DSingle_e1_l0.0001_b4_w24_s50000_r0.001_c_False_n_False_t1.0/trained_models/'

        main(data_path,
             lung_path,
             model_path,
             temperature,
             dataset_tag,
             threshold,
             step_lower=20000,
             step_upper=100000)




