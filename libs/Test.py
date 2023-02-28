import torch
# torch.manual_seed(0)
import errno
import numpy as np
# import pandas as pd
import os
from os import listdir
# import Image

import timeit

# import tifffile as tiff

# from skimage import exposure

from libs.Metrics import segmentation_scores


# def stitch_subvolumes():
# this function is to stich up all subvolume into the whole


def segment_whole_volume(model,
                         volume,
                         train_size=[192, 192, 192],
                         class_no=2):
    '''
    volume (numpy): c x d x h x w
    model: loaded model
    calculate iou for each subvolume then sum them up then average, don't ensemble the volumes in gpu
    '''
    c, d, h, w = np.shape(volume)
    no_d = d // train_size[0]
    no_h = h // train_size[1]
    no_w = w // train_size[2]
    segmentation = np.zeros_like(volume)
    # segmentation = torch.from_numpy(segmentation).to(device='cuda', dtype=torch.float32)

    # Loop through the whole volume:
    for i in range(0, no_d-1, 1):
        for j in range(0, no_h-1, 1):
            for k in range(0, no_w-1, 1):
                subvolume = volume[:, i*train_size[0]:(i+1)*train_size[0], j*train_size[1]:(j+1)*train_size[1], k*train_size[2]:(k+1)*train_size[2]]

                # plt.imshow(subvolume[0, 0, :, :])
                # plt.show()

                subvolume = torch.from_numpy(subvolume).to(device='cuda', dtype=torch.float32)
                subseg, _ = model(subvolume.unsqueeze(0))
                if class_no == 2:
                    subseg = torch.sigmoid(subseg)
                    # subseg = (subseg > 0.5).float()
                else:
                    subseg = torch.softmax(subseg, dim=1)
                    # _, subseg = torch.max(subseg, dim=1)

                # slice = subseg.detach().cpu().numpy()
                # plt.imshow(slice[0, 0, 0, :, :])
                # plt.show()

                segmentation[:, i*train_size[0]:(i+1)*train_size[0], j*train_size[1]:(j+1)*train_size[1], k*train_size[2]:(k+1)*train_size[2]] = subseg.detach().cpu().numpy()

    # corner case:
    subvolume = volume[:, d-train_size[0]:d, h-train_size[1]:h, w-train_size[2]:w]
    subvolume = torch.from_numpy(subvolume).to(device='cuda', dtype=torch.float32)

    subseg, _ = model(subvolume.unsqueeze(0))

    if class_no == 2:
        subseg = torch.sigmoid(subseg)
    else:
        subseg = torch.softmax(subseg, dim=1)

    segmentation[:, d-train_size[0]:d, h-train_size[1]:h, w-train_size[2]:w] = subseg.squeeze(0).detach().cpu().numpy()

    return segmentation


def ensemble_segmentation(model_path, volume, train_size=[192, 192, 192], class_no=2):
    segmentation = []
    all_models = [os.path.join(model_path, f) for f in listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
    all_models.sort()
    for i, model in enumerate(all_models):
        model = torch.load(model)
        model.eval()
        # with torch.no_grad:
        current_seg = segment_whole_volume(model, volume, train_size, class_no)
        segmentation.append(current_seg)
        # segmentation += current_seg

    segmentation = sum(segmentation) / len(segmentation)
    if class_no == 2:
        # segmentation = (segmentation > 0.5).float()
        segmentation = np.where(segmentation > 0.5, 1.0, 0.0)
    else:
        _, segmentation = torch.max(segmentation, dim=1)
    return segmentation


def test(saved_information_path,
         saved_model_path,
         test_data_path,
         test_label_path,
         device,
         model_name,
         class_no,
         size=[192, 192, 192],
         dilation=1):

    save_path = saved_information_path + '/results'
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    all_cases = [os.path.join(test_data_path, f) for f in listdir(test_data_path)]
    all_cases.sort()
    all_labels = [os.path.join(test_label_path, f) for f in listdir(test_label_path)]
    all_labels.sort()

    test_iou = []

    for each_case, each_label in zip(all_cases, all_labels):

        volume = np.load(each_case)
        label = np.load(each_label)

        label = torch.from_numpy(label).to(device='cuda', dtype=torch.float32)
        segmentation = ensemble_segmentation(saved_model_path, volume, train_size=size, class_no=class_no)
        # print(segmentation.size())
        test_mean_iu_ = segmentation_scores(label.squeeze(), segmentation.squeeze(), class_no)
        test_iou.append(test_mean_iu_)

        # stop = timeit.default_timer()
        # test_time = stop - start

    # print(len(testdata))
    result_dictionary = {
        'Test IoU mean': str(np.nanmean(test_iou)),
        'Test IoU std': str(np.nanstd(test_iou)),
        # 'Test IoU wt mean': str(np.nanmean(test_iou_wt)),
        # 'Test IoU wt std': str(np.nanstd(test_iou_wt)),
        # 'Test IoU et mean': str(np.nanmean(test_iou_et)),
        # 'Test IoU et std': str(np.nanstd(test_iou_et)),
        # 'Test IoU tc mean': str(np.nanmean(test_iou_tc)),
        # 'Test IoU tc std': str(np.nanstd(test_iou_tc)),
        # 'Test H-dist mean': str(np.nanmean(test_h_dist)),
        # 'Test H-dist std': str(np.nanstd(test_h_dist)),
        # 'Test recall mean': str(np.nanmean(test_recall)),
        # 'Test recall std': str(np.nanstd(test_recall)),
        # 'Test precision mean': str(np.nanmean(test_precision)),
        # 'Test precision std': str(np.nanstd(test_precision)),
        # 'Training time(s)': str(training_time),
        # 'Test time(s)': str(test_time / len(testdata))
    }

    ff_path = save_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()

    iou_path = save_path + '/iou.csv'
    # recall_path = save_path + '/recall.csv'
    # precision_path = save_path + '/precision.csv'

    np.savetxt(iou_path, test_iou, delimiter=',')
    # np.savetxt(recall_path, test_recall, delimiter=',')
    # np.savetxt(precision_path, test_precision, delimiter=',')

    return test_iou


def test_brats(saved_information_path, saved_model_path, testdata, device, model_name, class_no, training_time, dilation=16):

    save_path = saved_information_path + '/results'
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    all_models = [os.path.join(saved_model_path, f) for f in listdir(saved_model_path) if os.path.isfile(os.path.join(saved_model_path, f))]
    all_models.sort()

    # all_models = glob.glob(os.path.join(saved_model_path, '*.pt'))
    # print(all_models)

    # testing acc with main decoder:
    test_iou = []
    test_iou_wt = []
    test_iou_et = []
    test_iou_tc = []

    test_h_dist = []
    test_recall = []
    test_precision = []

    for model in all_models:
        # test_time = 0
        model = torch.load(model)
        model.eval()

        start = timeit.default_timer()
        with torch.no_grad():

            for ii, (test_images, test_label, test_imagename) in enumerate(testdata):
                test_img = test_images.to(device=device, dtype=torch.float32)
                # test_img = test_images.to(device=device, dtype=torch.float32).unsqueeze(1)
                test_label = test_label.to(device=device, dtype=torch.float32)

                assert torch.max(test_label) != 100.0

                if 'CCT' in model_name or 'cct' in model_name:
                    test_outputs, _ = model(test_img)
                elif 'expert' in model_name:
                    test_outputs = model(test_img, dilation)
                else:
                    test_outputs, _ = model(test_img)

                if class_no == 2:
                    test_class_outputs = torch.sigmoid(test_outputs)
                    test_class_outputs = (test_class_outputs > 0.5).float()
                else:
                    # _, test_class_outputs = torch.max(test_outputs, dim=1)
                    test_class_outputs = torch.sigmoid(test_outputs)
                    test_class_outputs = (test_class_outputs > 0.5).float()

                # testing on average metrics:
                # test_label = test_label.squeeze()
                # test_class_outputs = test_class_outputs.squeeze()

                # whole tumour: 1 == 1, 2 == 1, 3 == 1
                test_label_wt = torch.zeros_like(test_label)
                # test_class_outputs_wt = torch.zeros_like(test_class_outputs)
                test_label_wt[test_label == 1] = 1
                test_label_wt[test_label == 2] = 1
                test_label_wt[test_label == 3] = 1
                if len(test_class_outputs.size()) == 4:
                    test_class_outputs_wt = test_class_outputs[:, 0, :, :]
                else:
                    test_class_outputs_wt = test_class_outputs[:, :, 0, :, :]
                # test_class_outputs_wt[test_class_outputs == 1] = 1
                # test_class_outputs_wt[test_class_outputs == 2] = 1
                # test_class_outputs_wt[test_class_outputs == 3] = 1
                test_mean_iu_wt_ = segmentation_scores(test_label_wt.squeeze(), test_class_outputs_wt.squeeze(), 2)
                test_iou_wt.append(test_mean_iu_wt_)

                # tumour core: 3 == 1, 1 == 1, 2 == 0
                test_label_tc = torch.zeros_like(test_label)
                # test_class_outputs_tc = torch.zeros_like(test_class_outputs)
                test_label_tc[test_label == 3] = 1
                test_label_tc[test_label == 1] = 1
                test_label_tc[test_label == 2] = 0
                if len(test_class_outputs.size()) == 4:
                    test_class_outputs_tc = test_class_outputs[:, 1, :, :]
                else:
                    test_class_outputs_tc = test_class_outputs[:, :, 1, :, :]
                # test_class_outputs_tc[test_class_outputs == 3] = 1
                # test_class_outputs_tc[test_class_outputs == 1] = 1
                # test_class_outputs_tc[test_class_outputs == 2] = 0
                test_mean_iu_tc_ = segmentation_scores(test_label_tc.squeeze(), test_class_outputs_tc.squeeze(), 2)
                test_iou_tc.append(test_mean_iu_tc_)

                # enhancing tumour core: 3 == 1, 1 == 0, 2 == 0
                test_label_et = torch.zeros_like(test_label)
                # test_class_outputs_et = torch.zeros_like(test_class_outputs)
                test_label_et[test_label == 3] = 1
                test_label_et[test_label == 1] = 0
                test_label_et[test_label == 2] = 0
                if len(test_class_outputs.size()) == 4:
                    test_class_outputs_et = test_class_outputs[:, 2, :, :]
                else:
                    test_class_outputs_et = test_class_outputs[:, :, 2, :, :]

                # test_class_outputs_et[test_class_outputs == 3] = 1
                # test_class_outputs_et[test_class_outputs == 1] = 0
                # test_class_outputs_et[test_class_outputs == 2] = 0
                test_mean_iu_et_ = segmentation_scores(test_label_et.squeeze(), test_class_outputs_et.squeeze(), 2)
                test_iou_et.append(test_mean_iu_et_)

                test_mean_iu_ = (test_mean_iu_tc_ + test_mean_iu_et_ + test_mean_iu_wt_) / 3
                # test_mean_iu_ = segmentation_scores(test_label.squeeze(), test_class_outputs.squeeze(), 2)
                # test_mean_f1_, test_mean_recall_, test_mean_precision_ = f1_score(test_label.squeeze(), test_class_outputs.squeeze(), class_no)

                test_iou.append(test_mean_iu_)
                # test_recall.append(test_mean_recall_)
                # test_precision.append(test_mean_precision_)

                # if (test_class_outputs == 1).sum() > 1 and (test_label == 1).sum() > 1:
                #     t_dist_ = hd95(test_class_outputs.squeeze(), test_label.squeeze(), class_no)
                #     test_h_dist.append(t_dist_)

        stop = timeit.default_timer()
        test_time = stop - start

    # print(len(testdata))
    result_dictionary = {
        'Test IoU mean': str(np.nanmean(test_iou)),
        'Test IoU std': str(np.nanstd(test_iou)),
        'Test IoU wt mean': str(np.nanmean(test_iou_wt)),
        'Test IoU wt std': str(np.nanstd(test_iou_wt)),
        'Test IoU et mean': str(np.nanmean(test_iou_et)),
        'Test IoU et std': str(np.nanstd(test_iou_et)),
        'Test IoU tc mean': str(np.nanmean(test_iou_tc)),
        'Test IoU tc std': str(np.nanstd(test_iou_tc)),
        # 'Test H-dist mean': str(np.nanmean(test_h_dist)),
        # 'Test H-dist std': str(np.nanstd(test_h_dist)),
        # 'Test recall mean': str(np.nanmean(test_recall)),
        # 'Test recall std': str(np.nanstd(test_recall)),
        # 'Test precision mean': str(np.nanmean(test_precision)),
        # 'Test precision std': str(np.nanstd(test_precision)),
        'Training time(s)': str(training_time),
        'Test time(s)': str(test_time / len(testdata))
    }

    ff_path = save_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()

    iou_path = save_path + '/iou.csv'
    iou_tc_path = save_path + '/iou_tc.csv'
    iou_wt_path = save_path + '/iou_wt.csv'
    iou_et_path = save_path + '/iou_et.csv'

    # h_dist_path = save_path + '/h_dist.csv'
    # recall_path = save_path + '/recall.csv'
    # precision_path = save_path + '/precision.csv'

    np.savetxt(iou_path, test_iou, delimiter=',')
    # np.savetxt(iou_wt_path, test_iou_wt, delimiter=',')
    # np.savetxt(iou_tc_path, test_iou_tc, delimiter=',')
    # np.savetxt(iou_et_path, test_iou_et, delimiter=',')
    # np.savetxt(h_dist_path, test_h_dist, delimiter=',')
    # np.savetxt(recall_path, test_recall, delimiter=',')
    # np.savetxt(precision_path, test_precision, delimiter=',')

    return test_iou






