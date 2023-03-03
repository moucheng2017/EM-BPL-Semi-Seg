import torch
import sys
sys.path.append("..")
# torch.manual_seed(0)
import numpy as np
# import pandas as pd
import os
from os import listdir
# import Image

import nibabel as nib

from libs.Metrics import segmentation_scores


def segment_whole_volume(model,
                         volume,
                         train_size=(96, 96, 96), # d x h x w
                         class_no=2,
                         full_resolution=False):
    '''
    volume: c x d x h x w
    model: loaded model
    calculate iou for each subvolume then sum them up then average, don't ensemble the volumes in gpu
    '''
    # np.shape(volume): d x h x w
    if class_no == 2:
        class_no = 1
    segmentation = np.zeros((class_no, np.shape(volume)[-3], np.shape(volume)[-2], np.shape(volume)[-1]))
    model.eval()

    interval_d = train_size[0] // 4
    interval_h = train_size[1] // 4
    interval_w = train_size[2] // 4

    for i in range(0, np.shape(volume)[-3] - train_size[0], interval_d):
        for j in range(0, np.shape(volume)[-2] - train_size[1], interval_h):
            for k in range(0, np.shape(volume)[-1] - train_size[2], interval_w):
                if len(np.shape(volume)) == 3:
                    subvolume = volume[
                                i:i+train_size[0],
                                j:j+train_size[1],
                                k:k+train_size[2]]
                    subvolume = torch.from_numpy(subvolume).to(device='cuda', dtype=torch.float32)
                    subseg = model(subvolume.unsqueeze(0).unsqueeze(0))
                elif len(np.shape(volume)) == 4:
                    subvolume = volume[:,
                                i:i+train_size[0],
                                j:j+train_size[1],
                                k:k+train_size[2]]

                    subvolume = torch.from_numpy(subvolume).to(device='cuda', dtype=torch.float32)
                    subseg = model(subvolume.unsqueeze(0))
                else:
                    raise NotImplementedError

                if class_no == 2:
                    subseg = torch.sigmoid(subseg['segmentation'])
                else:
                    subseg = torch.softmax(subseg['segmentation'], dim=1)

                segmentation[:,
                i:i+train_size[0],
                j:j+train_size[1],
                k:k+train_size[2]] = subseg.squeeze().detach().cpu().numpy()

    # corner case the last one:
    if len(np.shape(volume)) == 4:
        subvolume = volume[:,
                    np.shape(volume)[-3]-train_size[0]:np.shape(volume)[-3],
                    np.shape(volume)[-2]-train_size[1]:np.shape(volume)[-2],
                    np.shape(volume)[-1]-train_size[2]:np.shape(volume)[-1]]

        subvolume = torch.from_numpy(subvolume).to(device='cuda', dtype=torch.float32)
        subseg = model(subvolume.unsqueeze(0))

    elif len(np.shape(volume)) == 3:
        subvolume = volume[
                    np.shape(volume)[-3] - train_size[0]:np.shape(volume)[-3],
                    np.shape(volume)[-2] - train_size[1]:np.shape(volume)[-2],
                    np.shape(volume)[-1] - train_size[2]:np.shape(volume)[-1]]

        subvolume = torch.from_numpy(subvolume).to(device='cuda', dtype=torch.float32)
        subseg = model(subvolume.unsqueeze(0).unsqueeze(0))

    else:
        raise NotImplementedError

    if class_no == 2:
        subseg = torch.sigmoid(subseg['segmentation'])
    else:
        subseg = torch.softmax(subseg['segmentation'], dim=1)

    segmentation[:,
    np.shape(volume)[-3] - train_size[0]:np.shape(volume)[-3],
    np.shape(volume)[-2] - train_size[1]:np.shape(volume)[-2],
    np.shape(volume)[-1] - train_size[2]:np.shape(volume)[-1]] = subseg.squeeze().detach().cpu().numpy()

    return segmentation.squeeze()


def segmentation_one_model(model_path,
                           data_path,
                           save_path,
                           transpose=1,
                           size=(64, 64, 64),
                           classno=2,
                           full_resolution=False,
                           save_flag=False):

    model = torch.load(model_path)
    model.eval()
    test_data_path = data_path + '/imgs'
    test_label_path = data_path + '/lbls'

    all_cases = [os.path.join(test_data_path, f) for f in listdir(test_data_path)]
    all_cases.sort()
    all_labels = [os.path.join(test_label_path, f) for f in listdir(test_label_path)]
    all_labels.sort()

    segmentation_iou_all_cases = []

    for each_case, each_label in zip(all_cases, all_labels):

        volume_nii = nib.load(each_case)
        volume = volume_nii.get_fdata()

        label_nii = nib.load(each_label)
        label = label_nii.get_fdata()

        # saved_segmentation = np.zeros_like(volume)

        if transpose > 0:
            if len(np.shape(volume)) == 3:
                volume = np.transpose(volume, (2, 0, 1))
                volume = np.expand_dims(volume, axis=0)
                label = np.transpose(label, (2, 0, 1))
            elif len(np.shape(volume)) == 4:
                volume = np.transpose(volume, (3, 2, 0, 1))
                label = np.transpose(label, (2, 0, 1))
            else:
                raise NotImplementedError

        # ensemble testing on different sizes
        segmentation_np = segment_whole_volume(model=model,
                                               volume=volume,
                                               train_size=size,
                                               class_no=classno,
                                               full_resolution=full_resolution)

        # segmentation_np = segment_whole_volume(model,
        #                                        volume,
        #                                        size,
        #                                        classno,
        #                                        full_resolution)

        if classno == 2:
            segmentation_np = np.where(segmentation_np > 0.5, 1, 0)
        else:
            segmentation_np = np.argmax(segmentation_np, axis=1)

        mean_iu = segmentation_scores(label.squeeze(), segmentation_np.squeeze(), classno)
        print(mean_iu)
        segmentation_iou_all_cases.append(mean_iu)
        # saved_segmentation = saved_segmentation*lung

        if save_flag is True:
            save_name_ext = os.path.split(each_case)[-1]
            save_name = os.path.splitext(save_name_ext)[0]
            save_name_nii = save_name + '_test_d' + str(size[0]) + '_r' + str(size[1]) + '.seg.nii.gz'
            segmentation_nii = nib.Nifti1Image(segmentation_np,
                                               volume_nii.affine,
                                               volume_nii.header)

            save_path_nii = os.path.join(save_path, save_name_nii)
            nib.save(segmentation_nii, save_path_nii)
            print(save_path_nii + ' is saved.\n')

    print('Test iou is: ' + str(sum(segmentation_iou_all_cases) / len(segmentation_iou_all_cases)))

    return segmentation_iou_all_cases


if __name__ == '__main__':
    # model_path = '/home/moucheng/PhD/2023_03_01_MIA/cluster_results/' \
    #              'Results_Task01_BrainTumour/cluster_brain/' \
    #              'Unet3D_l_0.0004_b1_w8_i40000_crop_d64_crop_h64_crop_w64/trained_models'
    # model_name = 'Unet3D_l_0.0004_b1_w8_i40000_crop_d64_crop_h64_crop_w64step21000.pt'

    model_path = '/home/moucheng/PhD/2023_03_01_MIA/cluster_results/Results_Task01_BrainTumour/cluster_brain/BPL3D_l_0.0004_b1_w8_d3_i40000_u2_mu0.5_thresh1_flag0_cd64_ch64_cw64/trained_models'
    model_name = 'BPL3D_l_0.0004_b1_w8_d3_i40000_u2_mu0.5_thresh1_flag0_cd64_ch64_cw64_best_train.pt'

    data_path = '/home/moucheng/projects_data/Task01_BrainTumour/test'

    save_path = '/home/moucheng/PhD/2023_03_01_MIA/results'

    segmentation_one_model(model_path=os.path.join(model_path, model_name),
                           data_path=data_path,
                           save_path=save_path,
                           transpose=1,
                           full_resolution=False,
                           size=(96, 128, 128),
                           classno=2,
                           save_flag=False)


# def test_all_models(model_path,
#                     data_path,
#                     save_path,
#                     size,
#                     classno=2,
#                     full_resolution=False,
#                     save_flag=False):
#     all_models = [os.path.join(model_path, f) for f in listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
#     all_models.sort()
#     for i, model in enumerate(all_models):
#         iou_one_model = segmentation_one_case_one_model(model, data_path, save_path, size, classno, full_resolution, save_flag)
#         if i == 0:
#             iou = iou_one_model
#         else:
#             iou = [value1+value2 for value1, value2 in zip(iou, iou_one_model)]
#     iou = [each_value / len(all_models) for each_value in iou]
#     assert len(iou) == len(iou_one_model)
#     mean_iou_all_models = np.nanmean(iou)
#     std_iou_all_models = np.nanstd(iou)
#     result_dictionary = {
#         'Test IoU mean': str(np.nanmean(iou)),
#         'Test IoU std': str(np.nanstd(iou))
#     }
#     ff_path = save_path + '/test_result_data.txt'
#     ff = open(ff_path, 'w')
#     ff.write(str(result_dictionary))
#     ff.close()
#     iou_path = save_path + '/iou.csv'
#     np.savetxt(iou_path, iou, delimiter=',')
#     return mean_iou_all_models, std_iou_all_models










