import os
import csv
import numpy as np
import pathlib
import scipy
import math
import sys
import argparse
import nibabel as nib
import torch
from pathlib import Path
sys.path.append('..')
import matplotlib.pyplot as plt
from skimage.transform import resize
from libs.Metrics import segmentation_scores

# This prints out a grid: rows are different images, each row has three columns of image, label, and overlapped image

parser = argparse.ArgumentParser('Run inference on BRATS')
parser.add_argument('--img_source', type=str, help='source image file', default='/home/moucheng/projects_data/Task01_BrainTumour/test/imgs')
parser.add_argument('--lbl_source', type=str, help='source label file', default='/home/moucheng/projects_data/Task01_BrainTumour/test/lbls')

# BPL
# parser.add_argument('--model_source', type=str, help='model path', default='/home/moucheng/PhD/2023_03_01_MIA/cluster_results/Results_Task01_BrainTumour/cluster_brain/BPL3D_l_0.0004_b1_w8_d3_i40000_u2_mu0.5_thresh1_flag0_cd64_ch64_cw64/trained_models/BPL3D_l_0.0004_b1_w8_d3_i40000_u2_mu0.5_thresh1_flag0_cd64_ch64_cw64_best_train.pt')
parser.add_argument('--model_source', type=str, help='model path', default='/home/moucheng/PhD/2023_03_01_MIA/cluster_results/Results_Task01_BrainTumour/cluster_brain/BPL3D_l_0.0004_b1_w8_d3_i40000_u2_mu0.5_thresh1_flag0_cd64_ch64_cw64/trained_models/BPL3D_l_0.0004_b1_w8_d3_i40000_u2_mu0.5_thresh1_flag0_cd64_ch64_cw64step23000.pt')

# Baseline
# parser.add_argument('--model_source', type=str, help='model path', default='/home/moucheng/PhD/2023_03_01_MIA/cluster_results/Results_Task01_BrainTumour/cluster_brain/Unet3D_l_0.0004_b1_w8_i40000_crop_d64_crop_h64_crop_w64/trained_models/Unet3D_l_0.0004_b1_w8_i40000_crop_d64_crop_h64_crop_w64_best_train.pt')
# parser.add_argument('--model_source', type=str, help='model path', default='/home/moucheng/PhD/2023_03_01_MIA/cluster_results/Results_Task01_BrainTumour/cluster_brain/Unet3D_l_0.0004_b1_w8_i40000_crop_d64_crop_h64_crop_w64/trained_models/Unet3D_l_0.0004_b1_w8_i40000_crop_d64_crop_h64_crop_w64step23000.pt')


parser.add_argument('--save_path', type=str, help='save path', default='/home/moucheng/PhD/2023_03_01_MIA/results')
parser.add_argument('--new_dim_d', type=int, help='new dimension d', default=128)
parser.add_argument('--new_dim_h', type=int, help='new dimension h', default=128)
parser.add_argument('--new_dim_w', type=int, help='new dimension w', default=128)
parser.add_argument('--transpose', type=str, help='transpose flag', default=1)
parser.add_argument('--test_cases', type=int, help='number of testing cases', default=400)
args = parser.parse_args()


if __name__ == '__main__':

    all_cases = [os.path.join(args.img_source, f) for f in os.listdir(args.img_source)]
    all_cases.sort()
    all_cases = all_cases[:args.test_cases]

    all_labels = [os.path.join(args.lbl_source, f) for f in os.listdir(args.lbl_source)]
    all_labels.sort()
    all_labels = all_labels[:args.test_cases]

    segmentation_iou_all_cases = []

    for case_index, (each_case, each_label) in enumerate(zip(all_cases, all_labels)):

        img_volume = nib.load(each_case)
        img_volume = img_volume.get_fdata()
        img = np.asfarray(img_volume)
        # print(np.shape(img))
        # img = (img - np.nanmean(img) + 1e-10) / (np.nanstd(img) + 1e-10)

        lbl_volume = nib.load(each_label)
        lbl_volume = lbl_volume.get_fdata()
        lbl = np.asfarray(lbl_volume)
        # print(np.shape(lbl))

        if args.transpose > 0:
            if len(np.shape(img)) == 3:
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)
                lbl = np.transpose(lbl, (2, 0, 1))
            elif len(np.shape(img)) == 4:
                img = np.transpose(img, (3, 2, 0, 1))
                lbl = np.transpose(lbl, (2, 0, 1))
            else:
                raise NotImplementedError

        Path(args.save_path).mkdir(parents=True, exist_ok=True)

        model = torch.load(args.model_source).cuda()

        # new_dim = args.new_dim

        if case_index == 0:
            print(np.shape(img))

        division_d = np.shape(img)[-3] // args.new_dim_d
        division_h = np.shape(img)[-2] // args.new_dim_h
        division_w = np.shape(img)[-1] // args.new_dim_w

        d_start = (np.shape(img)[-3] - division_d*args.new_dim_d) // 2
        d_end = d_start + division_d*args.new_dim_d

        h_start = (np.shape(img)[-2] - division_h*args.new_dim_h) // 2
        h_end = h_start + division_h*args.new_dim_h

        w_start = (np.shape(img)[-1] - division_w*args.new_dim_w) // 2
        w_end = w_start + division_w*args.new_dim_w

        img = img[:, d_start:d_end, h_start:h_end, w_start:w_end]
        lbl = lbl[d_start:d_end, h_start:h_end, w_start:w_end]

        if case_index == 0:
            print(np.shape(img))

        seg = np.zeros((np.shape(img)[-3], np.shape(img)[-2], np.shape(img)[-1]))

        imgs_d = np.split(img, np.shape(img)[-3] // args.new_dim_d, axis=1)

        count = 0
        # print('start to segment each sub volume...')

        for i, each_img_d in enumerate(imgs_d):
            imgs_d_h = np.split(each_img_d, np.shape(img)[-2] // args.new_dim_h, axis=2)
            for j, each_img_h in enumerate(imgs_d_h):
                imgs_d_h_w = np.split(each_img_h, np.shape(img)[-1] // args.new_dim_w, axis=3)
                for k, each_img_w in enumerate(imgs_d_h_w):

                    assert np.shape(each_img_w)[-3] == args.new_dim_d
                    assert np.shape(each_img_w)[-2] == args.new_dim_h
                    assert np.shape(each_img_w)[-1] == args.new_dim_w

                    seg_ = torch.from_numpy(each_img_w).cuda().unsqueeze(0).float()
                    seg_ = model(seg_)
                    seg_ = seg_.get('segmentation')
                    seg_ = torch.sigmoid(seg_)
                    seg_ = (seg_ > 0.5).float()

                    seg_ = seg_.squeeze().detach().cpu().numpy()
                    seg[
                    i*args.new_dim_d:(i+1)*args.new_dim_d,
                    j*args.new_dim_h:(j+1)*args.new_dim_h,
                    k*args.new_dim_w:(k+1)*args.new_dim_w] = seg_

                    count += 1
                    # print(count)

                    del each_img_w
                    del seg_

        mean_iu = segmentation_scores(lbl.squeeze(), seg.squeeze(), 2)
        print(mean_iu)
        segmentation_iou_all_cases.append(mean_iu)

    # seg = nib.Nifti1Image(seg, affine=np.eye(4))
    # seg_name = str(args.flag) + '_d' + str(args.new_dim) + '_c' + str(args.confidence) + '_segmentation.nii'
    # save_file = os.path.join(args.save_path, seg_name)
    # nib.save(seg, save_file)

    print('\n')
    print('\n')
    print('Test iou mean is: ' + str(sum(segmentation_iou_all_cases) / len(segmentation_iou_all_cases)))
    print('Test iou std is: ' + str(np.nanstd(segmentation_iou_all_cases)))
    print('Done')




