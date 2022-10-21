import torch
import sys
sys.path.append("../..")
# torch.manual_seed(0)
import numpy as np
# import pandas as pd
import os
from os import listdir
# import Image

import nibabel as nib

from Metrics import segmentation_scores


def segment_whole_volume(model,
                         volume,
                         train_size,
                         class_no=2,
                         full_resolution=False):
    '''
    volume: c x d x h x w
    model: loaded model
    calculate iou for each subvolume then sum them up then average, don't ensemble the volumes in gpu
    '''
    # print(np.shape(volume))
    c, d, h, w = np.shape(volume)
    volume[volume < -1000.0] = -1000.0
    volume[volume > 500.0] = 500.0
    volume = (volume - np.nanmean(volume)) / np.nanstd(volume)
    # volume = RandomContrast(bin_range=[100, 150, 200]).randomintensity(volume)
    segmentation = np.zeros_like(volume)
    model.eval()

    ratio_h = 10
    # Loop through the whole volume:

    if full_resolution is False:
        for i in range(0, d - train_size[0], train_size[0]//2):
            for j in range(0, h - train_size[1], train_size[1]//2):
                for k in range(0, w - train_size[2], train_size[2]//2):
                    subvolume = volume[:, i:i+train_size[0], j:j+train_size[1], k:k+train_size[2]]
                    subvolume = (subvolume - np.nanmean(subvolume)) / np.nanstd(subvolume)
                    subvolume = torch.from_numpy(subvolume).to(device='cuda', dtype=torch.float32)
                    subseg, _ = model(subvolume.unsqueeze(0))

                    if class_no == 2:
                        subseg = torch.sigmoid(subseg)
                        # subseg = (subseg > 0.5).float()
                    else:
                        subseg = torch.softmax(subseg, dim=1)
                        # _, subseg = torch.max(subseg, dim=1)

                    segmentation[:, i:i+train_size[0], j:j+train_size[1], k:k+train_size[2]] = subseg.detach().cpu().numpy()
    else:
        for i in range(0, d - train_size[0] - 1, 2):
            subvolume = volume[:, i:i + train_size[0], :, :]
            subvolume = (subvolume - np.nanmean(subvolume)) / np.nanstd(subvolume)
            subvolume = torch.from_numpy(subvolume).to(device='cuda', dtype=torch.float32)
            subseg = model(subvolume.unsqueeze(0))

            if class_no == 2:
                subseg = torch.sigmoid(subseg)
                # subseg = (subseg > 0.5).float()
            else:
                subseg = torch.softmax(subseg, dim=1)
                # _, subseg = torch.max(subseg, dim=1)

            segmentation[:, i:i + train_size[0], :, :] = subseg.detach().cpu().numpy()

    # corner case the last one:
    subvolume = volume[:, d-train_size[0]:d, h-train_size[1]:h, w-train_size[2]:w]
    subvolume = (subvolume - np.nanmean(subvolume)) / np.nanstd(subvolume)
    subvolume = torch.from_numpy(subvolume).to(device='cuda', dtype=torch.float32)
    subseg, _ = model(subvolume.unsqueeze(0))

    if class_no == 2:
        subseg = torch.sigmoid(subseg)
        # subseg = (subseg > 0.9).float()
    else:
        subseg = torch.softmax(subseg, dim=1)
        # _, subseg = torch.max(subseg, dim=1)
    segmentation[:, d-train_size[0]:d, h-train_size[1]:h, w-train_size[2]:w] = subseg.squeeze(0).detach().cpu().numpy()
    return segmentation


def segmentation_one_case_one_model(model_path,
                                    data_path,
                                    save_path,
                                    size,
                                    classno=2,
                                    full_resolution=False,
                                    save_flag=False):

    model = torch.load(model_path)
    test_data_path = data_path + '/imgs'
    test_label_path = data_path + '/lbls'
    test_lung_path = data_path + '/lung'

    all_cases = [os.path.join(test_data_path, f) for f in listdir(test_data_path)]
    all_cases.sort()
    all_labels = [os.path.join(test_label_path, f) for f in listdir(test_label_path)]
    all_labels.sort()
    all_lungs = [os.path.join(test_lung_path, f) for f in listdir(test_lung_path)]
    all_lungs.sort()

    segmentation_iou_all_cases = []

    for each_case, each_label, each_lung in zip(all_cases, all_labels, all_lungs):

        # print(each_case)
        volume_nii = nib.load(each_case)
        volume = volume_nii.get_fdata()

        label_nii = nib.load(each_label)
        label = label_nii.get_fdata()

        lung_nii = nib.load(each_lung)
        lung = lung_nii.get_fdata()

        saved_segmentation = np.zeros_like(volume)

        volume = np.transpose(volume, (2, 0, 1))
        volume = np.expand_dims(volume, axis=0)

        save_name_ext = os.path.split(each_case)[-1]
        save_name = os.path.splitext(save_name_ext)[0]
        # save_name_nii = save_name + '_seg.nii.gz'
        save_name_nii = save_name + '_test_d' + str(size[0]) + '_r' + str(size[1]) + '.seg.nii.gz'
        # segmentation_np = None

        # ensemble testing on different sizes
        # for each_size in sizes:
        segmentation_np = segment_whole_volume(model,
                                               volume,
                                               size,
                                               classno,
                                               full_resolution)

        segmentation_np = segmentation_np[0, :, :, :]
        # segmentation_np = np.transpose(segmentation_np, (1, 2, 0))
        # segmentation_np += segmentation_np_current

        # segmentation_np = segmentation_np / len(sizes)

        if classno == 2:
            segmentation_np = np.where(segmentation_np > 0.5, 1, 0)
        else:
            segmentation_np = np.argmax(segmentation_np, axis=1)

        h, w, d = np.shape(saved_segmentation)

        # print(np.shape(segmentation_np))
        # print(np.shape(saved_segmentation))

        for dd in range(d):
            saved_segmentation[:, :, dd] = segmentation_np[dd, :, :]

        mean_iu, _, __ = segmentation_scores(label, saved_segmentation, classno)
        segmentation_iou_all_cases.append(mean_iu)

        saved_segmentation = saved_segmentation*lung

        if save_flag is True:
            segmentation_nii = nib.Nifti1Image(saved_segmentation,
                                               volume_nii.affine,
                                               volume_nii.header)

            save_path_nii = os.path.join(save_path, save_name_nii)
            nib.save(segmentation_nii, save_path_nii)
            print(save_path_nii + ' is saved.\n')

    return segmentation_iou_all_cases


def test_all_models(model_path,
                       data_path,
                       save_path,
                       size,
                       classno=2,
                       full_resolution=False,
                       save_flag=False):

    all_models = [os.path.join(model_path, f) for f in listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
    all_models.sort()

    for i, model in enumerate(all_models):
        iou_one_model = segmentation_one_case_one_model(model, data_path, save_path, size, classno, full_resolution, save_flag)
        if i == 0:
            iou = iou_one_model
        else:
            iou = [value1+value2 for value1, value2 in zip(iou, iou_one_model)]

    iou = [each_value / len(all_models) for each_value in iou]
    assert len(iou) == len(iou_one_model)
    mean_iou_all_models = np.nanmean(iou)
    std_iou_all_models = np.nanstd(iou)

    result_dictionary = {
        'Test IoU mean': str(np.nanmean(iou)),
        'Test IoU std': str(np.nanstd(iou))
    }

    ff_path = save_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()

    iou_path = save_path + '/iou.csv'
    np.savetxt(iou_path, iou, delimiter=',')

    return mean_iou_all_models, std_iou_all_models


if __name__ == '__main__':
    model_path = '/home/moucheng/PhD/2022_12_Clinical/20220511/Results/airway/airway_balanced/' \
                 'Sup3D_e1_l0.001_b2_w64_s5000_d3_r0.01_z8_x480/trained_models/'

    model_name = 'Sup3D_e1_l0.001_b2_w64_s5000_d3_r0.01_z8_x480_4996.pt'

    data_path = '/home/moucheng/projects_data/Pulmonary_data/airway/test'

    save_path = '/home/moucheng/PhD/2022_12_Clinical/segmentation_20220511'

    segmentation_one_case_one_model(os.path.join(model_path, model_name),
                                    data_path,
                                    save_path,
                                    size=[8, 480, 480],
                                    classno=2,
                                    save_flag=True)












