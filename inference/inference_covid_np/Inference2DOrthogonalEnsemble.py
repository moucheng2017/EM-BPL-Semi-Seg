import nibabel

import sys
sys.path.append("..")
sys.path.append("../..")

import torch
import numpy as np
import os

import numpy.ma as ma

from libs.Augmentations import RandomContrast, norm95

from libs.PTKAirways import RunPTKAirways
from skimage.measure import label
import raster_geometry as rsg
from scipy.ndimage import binary_closing


def nii2np(file_path):
    # Read image:
    data = nibabel.load(file_path)
    data = data.get_fdata()
    data = np.array(data, dtype="float32")
    # print(np.shape(data))
    # Now applying lung window:
    data[data < -1000.0] = -1000.0
    data[data > 500.0] = 500.0
    # H X W X D --> D X H X W:
    data = np.transpose(data, (2, 0, 1))
    d, h, w = np.shape(data)
    # print(np.shape(data))
    data = np.pad(
        data, pad_width=((0, 512 - d), (0, 512 - h), (0, 512 - w)), mode="symmetric"
    )

    if len(np.shape(data)) == 3:
        data = np.expand_dims(data, axis=0)  # 1 X D X H X W
    return data, h, w, d


def np2tensor(data):
    data = torch.from_numpy(data).to(device="cuda", dtype=torch.float32)
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


def adjustcontrast(data, lung_mask, adjust_times=0):
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


def seg_one_plane(volume, model, direction=0, temperature=2):

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
            img = volume[:, i, :, :]  # B x 1 x 512 x 512
            img = img.unsqueeze(1)
        elif direction == 1:
            img = volume[:, :, i, :]  # B x 512 x 1 x 512
            img = img.unsqueeze(1)
        elif direction == 2:
            img = volume[:, :, :, i]  # B x 512 x 512 x 1
            img = img.unsqueeze(1)

        # print(img.size())
        seg_, _ = model(img)
        seg_ = torch.sigmoid(seg_ / temperature)
        seg_ = seg_.squeeze().detach().cpu().numpy()  # W x D x H

        if direction == 0:
            seg[i, :, :] = seg_
        elif direction == 1:
            seg[:, i, :] = seg_
        elif direction == 2:
            seg[:, :, i] = seg_

    return seg


def seg_three_plaines(volume, model, lung, temperature=2):

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

    seg = seg * lung
    del lung

    return seg


def save_seg(save_path, save_name, nii_path, saved_data):

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    nii = nibabel.load(nii_path)
    segmentation_nii = nibabel.Nifti1Image(saved_data, nii.affine, nii.header)
    save_path_nii = os.path.join(save_path, save_name)
    nibabel.save(segmentation_nii, save_path_nii)


def ensemble(seg_path, threshold):

    final_seg = []
    all_segs = os.listdir(seg_path)
    all_segs.sort()
    all_segs = [
        os.path.join(seg_path, seg_name) for seg_name in all_segs if "prob" in seg_name
    ]

    for seg_path in all_segs:
        seg = nib.load(seg_path)
        seg = seg.get_fdata()
        seg = np.array(seg, dtype="float32")
        final_seg.append(seg)
        os.remove(seg_path)

    output = sum(final_seg) / len(final_seg)
    output = np.where(output > threshold, 1, 0)
    del final_seg
    return output


def ptkseg(model_out, data, spacings):
    '''
    Add PTK Airway segmentation, originally by Tom Doel, translated to python by
    Adam Szmul and maintained by Ashkan Pakzad.

    A region growing algorithm that has a strict criteria on how much it can grow
    every iteration. It automatically identifies the trachea and seeds from there.
    It propogates waves downwards, splitting onto multiple fronts as the airways
    split.
    '''
    smooth = True
    air_threshold = -775
    maximum_number_of_generations = 20
    explosion_multiplier = 3  # lower = more conservative
    PTKAirwaylabel = RunPTKAirways(
        data,
        spacings,
        smooth=smooth,
        air_threshold=air_threshold,
        maximum_number_of_generations=maximum_number_of_generations,
        explosion_multiplier=explosion_multiplier,
    )
    return np.logical_or(model_out, PTKAirwaylabel)

def morphoclose(seg, data):
    '''
    Morphological closing of input segmentation using square 3x3 structuring
    element and strict voxel threshold criteria. The result will close the
    existing segmentation but only if new voxels are within the threshold
    criteria.
    '''
    air_threshold = -775
    # identify threshold voxels
    airvox = data < air_threshold;

    # morpho opening
    se = np.array(rsg.square(4, 3)).astype(int)

    # morpho closing
    y_im = seg.copy()
    x_im = seg.copy()

    for yi in range(y_im.shape[0]):
        y_im[yi,:,:] = binary_closing(seg[yi,:,:].squeeze(),structure=se)
    for xi in range(x_im.shape[1]):
        x_im[:,xi,:] = binary_closing(seg[:,xi,:].squeeze(),structure=se)

    # init closed image
    closed_im = np.logical_or(y_im, x_im)

    # new voxels by closing MUST be air voxels by threshold
    difference = np.logical_and(airvox, (closed_im.astype(int) - seg.astype(int)).astype(bool))
    # add new confirmed airway voxels to union segmentation
    closed_im = np.logical_or(seg, difference)
    return np.logical_or(seg, closed_im)

def largestcc(seg):
    '''
    Extracts the largest connected component in a segmentation mask
    '''
    labels = label(seg)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def main(test_data_path,
         case,
         lung_path,
         model_path,
         temp,
         channel=32,
         threshold=0.8,
         step_lower=20000,
         step_upper=20200):

    # generate save path:
    path = Path(os.path.abspath(model_path))
    save_path_parent = path.parent.absolute()

    save_path = os.path.join(
        save_path_parent, "segmentation_thresh" + str(threshold) + "_temp" + str(temp)
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path_parent, 'segmentation_thresh' + str(threshold) + '_temp' + str(temp))
    save_path = os.path.join(save_path, case)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # sort out all models:
    all_models = os.listdir(model_path)
    all_models.sort()

    # ran inference_covid for each model:
    for model_name in all_models:

        step = model_name.split("_")[-1]
        step = float(step.split(".")[0])

        if step_lower < step < step_upper:
            model_name = os.path.join(model_path, model_name)

            # nii --> np:
            data, d1, h1, w1 = nii2np(test_data_path)
            lung, d2, _, _ = nii2np(lung_path)
            assert d1 == d2

            # load model:
            # print(model_name)
            # model = torch.load(model_name)
            # model.load_state_dict()
            # print(model)

            model = Unet2DMultiChannel(in_ch=1, width=channel, output_channels=1)
            model.to('cuda')
            checkpoint = torch.load(model_name,map_location='cuda:0')
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
            save_name = case + "_s" + str(step) + "_prob.nii.gz"

            # save seg:
            save_seg(save_path, save_name, test_data_path, seg)

    # ensemble all segmentation files:
    seg = ensemble(save_path, threshold)

    # add post process PTK airways
    data = np.transpose(data, (1, 2, 0))
    seg = ptkseg(seg, data, (h1, w1, d1))

    # post process morphological closing
    seg = morphoclose(seg, data)

    # extract largest CC
    final_seg = largestcc(seg)

    save_name = case + "_final_prob.nii.gz"

    # save seg:
    save_seg(save_path, save_name, test_data_path, final_seg)


if __name__ == "__main__":
    # Hyperparameters:
    cases = ["6357B"]  # testing case
    threshold = 0.4  # confidence threshold
    temperature = 2  # temperature scaling for sigmoid/softmax

    for case in cases:
        data_path = (
            "/home/moucheng/projects_data/Pulmonary_data/airway/inference/imgs/"
            + case
            + ".nii.gz"
        )
        lung_path = (
            "/home/moucheng/projects_data/Pulmonary_data/airway/inference/lung/"
            + case
            + "_lunglabel.nii.gz"
        )
        model_path = "/home/moucheng/projects_codes/Results/airway/2022_07_04/OrthogonalSup2DSingle_e1_l0.0001_b4_w24_s50000_r0.001_c_False_n_False_t1.0/trained_models/"

        main(
            data_path,
            case,
            lung_path,
            model_path,
            temperature,
            threshold,
            step_lower=20000,
            step_upper=100000,
        )
