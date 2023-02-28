import nibabel
import sys
sys.path.append("..")
sys.path.append("../..")
from skimage.transform import resize
import torch
import numpy as np
import os
from libs.Augmentations import RandomContrast, norm95

# Work flow for each case:
# 1. Read nii.gz
# 1.5. Prepare an empty volume for storing slices
# 2. Transform nii.gz into np
# 3. Slice np file
# 4. Expand the dim of the slice for inference_covid
# 5. Segmentation
# 6. Store them in NIFTI file format


def nii2np(file_path):
    # Read image:
    data = nibabel.load(file_path)
    data = data.get_fdata()
    data = np.array(data, dtype='float32')
    print(np.shape(data))
    # Now applying lung window:
    data[data < -1000.0] = -1000.0
    data[data > 500.0] = 500.0
    # H X W X D --> D X H X W:
    data = np.transpose(data, (2, 0, 1))
    if len(np.shape(data)) == 3:
        data = np.expand_dims(data, axis=0) # 1 X D X H X W
    return data


def adjustcontrast(data,
                   adjust_times=0):
    outputs = []
    if adjust_times == 0:
        data = norm95(data)
        return outputs.append(data)
    else:
        contrast_augmentation = RandomContrast(bin_range=[10, 250])
        for i in range(adjust_times):
            data_ = contrast_augmentation.randomintensity(data)
            data_ = norm95(data_)
            outputs.append(data_)
        return outputs


def seg_d_direction(volume,
                    model,
                    temperature=2,
                    interval=10):

    d, h, w = np.shape(volume)
    # print(np.shape(volume))
    print('volume has ' + str(d) + ' d slices')
    seg = np.zeros((d, 160, 160))
    for dd in range(0, d, 1):
        img = volume[dd, :, :].squeeze()
        img = resize(img, (160, 160))
        img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
        img = torch.from_numpy(img).to(device='cuda', dtype=torch.float32)
        seg_patch = model(img)
        seg_patch = seg_patch.get('segmentation')
        seg_patch = torch.sigmoid(seg_patch / temperature)
        seg_patch = seg_patch.squeeze().detach().cpu().numpy()  # H x D x W
        seg[dd, :, :] = seg_patch
        print('d plane slice ' + str(dd) + ' is done...')
    return seg


def seg_h_direction(volume,
                    model,
                    temperature=2,
                    interval=10
                    ):

    d, h, w = np.shape(volume)
    # print(np.shape(volume))
    print('volume has ' + str(h) + ' h slices')
    seg = np.zeros((d, 160, 160))
    for hh in range(0, 160, 1):
        img = volume[:, hh, :].squeeze()
        for d_ in range(0, d-161, interval):
            cropped = img[d_:d_+160, :].squeeze()
            cropped = resize(cropped, (160, 160))
            cropped = np.expand_dims(np.expand_dims(cropped, axis=0), axis=0)
            cropped = torch.from_numpy(cropped).to(device='cuda', dtype=torch.float32)
            seg_patch = model(cropped)
            seg_patch = seg_patch.get('segmentation')
            seg_patch = torch.sigmoid(seg_patch / temperature)
            seg_patch = seg_patch.squeeze().detach().cpu().numpy()
            seg[d_:d_+160, hh, :] = seg_patch
        print('h plane slice ' + str(hh) + ' is done...')

    return seg


def seg_w_direction(volume,
                    model,
                    temperature=2,
                    interval=160
                    ):

    d, h, w = np.shape(volume)
    print('volume has ' + str(w) + ' w slices')
    seg = np.zeros((d, 160, 160))
    for ww in range(0, 160, 1):
        img = volume[:, :, ww].squeeze()
        for d_ in range(0, d-161, interval):
            cropped = img[d_:d_+160, :].squeeze()
            cropped = resize(cropped, (160, 160))
            cropped = np.expand_dims(np.expand_dims(cropped, axis=0), axis=0)
            cropped = torch.from_numpy(cropped).to(device='cuda', dtype=torch.float32)
            seg_patch = model(cropped)
            seg_patch = seg_patch.get('segmentation')
            seg_patch = torch.sigmoid(seg_patch / temperature)
            seg_patch = seg_patch.squeeze().detach().cpu().numpy()
            seg[d_:d_+160, :, ww] = seg_patch
            # print()
            # print('4')
        print('w plane slice ' + str(ww) + ' is done...')

    return seg


def merge_segs(folder):
    all_files = os.listdir(folder)
    seg = 0.0
    for each_seg in all_files:
        each_seg = np.load(each_seg)
        seg += each_seg

    seg = seg / len(all_files)
    seg = (seg > 0.9).float()
    return seg


def segment2D(test_data_path,
              model_path,
              threshold,
              temperature,
              contrast=(10, 20)):

    # nii --> np:
    # data = nii2np(test_data_path)
    # d, h, w = np.shape(data)
    # print(np.shape(data))

    data = np.load(test_data_path)
    d, h, w = np.shape(data)
    data = resize(data, (d, 160, 160))

    # Use different contrast:
    contrast_augmentation = RandomContrast(bin_range=[contrast[0], contrast[1]])
    data = contrast_augmentation.randomintensity(data)

    # Load model:
    model = torch.load(model_path)
    model.cuda()
    model.eval()

    output_w = seg_w_direction(data, model, temperature)
    output_h = seg_h_direction(data, model, temperature)
    output_d = seg_d_direction(data, model, temperature)
    output_prob = (output_d + output_h + output_w) / 3
    output = np.where(output_prob > threshold, 1, 0)

    return np.squeeze(output), np.squeeze(output_prob)


def ensemble(seg_path):
    all_segs = os.listdir(seg_path)
    all_segs.sort()
    all_segs = [os.path.join(seg_path, seg_name) for seg_name in all_segs if 'prob' in seg_name]
    final_seg = 0.0
    for seg in all_segs:
        seg = np.load(seg)
        final_seg += seg
    return final_seg / len(all_segs)


def save_seg(save_path,
             save_name,
             saved_data):

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    save_path_full = os.path.join(save_path, save_name)
    np.save(save_path_full, saved_data)


if __name__ == "__main__":
    case = '2'
    threshold = 0.99
    temperature = 2.0
    contrast = 200

    save_path = '/home/moucheng/Results_class1.0/PPFE/'
    data_path = '/home/moucheng/projects_data/PPFE_HipCT/processed/imgs/' + case + 'img.npy'

    # save_path = '/home/moucheng/Results_class1.0/'
    # data_path = '/home/moucheng/projects_data/COVID_ML_data/COVID-CNN/validation_dataset/stack_' + case + '.npy'

    model_path = '/home/moucheng/Results_class1.0/2.5D/Unet_l0.0005_b16_w32_d4_i200000_l2_0.01_c_1_t2.0/trained_models/'
    model_name = 'Unet_l0.0005_b16_w32_d4_i200000_l2_0.01_c_1_t2.0_best_val.pt'
    model_path_full = model_path + model_name

    save_name = case + '_seg_2Dorthogonal_thresh' + str(threshold) + '_temp' + str(temperature) + '_ctr' + str(contrast) + '.npy'
    save_name_prob = case + '_prob_2Dorthogonal_thresh' + str(threshold) + '_temp' + str(temperature) + '_ctr' + str(contrast) + '.npy'

    segmentation, probability = segment2D(data_path,
                                          model_path_full,
                                          threshold,
                                          temperature,
                                          (contrast, contrast))

    save_seg(save_path, save_name, segmentation)
    save_seg(save_path, save_name_prob, probability)

    print(np.shape(segmentation))
    print('End')
