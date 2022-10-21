import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from scipy.ndimage import _ni_support

from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy import ndimage
# good references:
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/niftynet/utilities/util_common.py
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/niftynet/evaluation/pairwise_measures.py
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/utils/metrics.py
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
#-----------------------------------------------


# def iou_metric(outputs, labels):
#     # ========================================================================================================
#     # method 2
#     # adopted from CSAILVision:
#     zeros_outputs = torch.zeros_like(outputs)
#     ones_outputs = torch.ones_like(outputs)
#     outputs = torch.where((outputs > 0.5), ones_outputs, zeros_outputs)
#     # outputs = (outputs > 0.5)
#     imPred = np.asarray(outputs.cpu().detach()).copy()
#     imLab = np.asarray(labels.cpu().detach()).copy()
#     imPred += 1
#     imLab += 1
#     # Remove classes from unlabeled pixels in gt image.
#     # We should not penalize detections in unlabeled portions of the image.
#     imPred = imPred * (imLab > 0)
#     # Compute area intersection:
#     intersection = imPred * (imPred == imLab)
#     (area_intersection, _) = np.histogram(
#         intersection, bins=2, range=(1, 2))
#     # Compute area union:
#     (area_pred, _) = np.histogram(imPred, bins=2, range=(1, 2))
#     (area_lab, _) = np.histogram(imLab, bins=2, range=(1, 2))
#     area_union = area_pred + area_lab - area_intersection
#     iou = area_intersection / (area_union+1e-8)
#     return iou.mean()
# ==============================================================================================================
# accuracy:


def preprocessing_accuracy(label_true, label_pred, n_class):

    # print(type(label_pred).__module__)
    # print(type(label_true))

    if type(label_pred).__module__ == np.__name__:
        label_pred = np.asarray(label_pred, dtype='int32')
    else:
        if n_class == 2:
        # thresholding predictions:
            output_zeros = torch.zeros_like(label_pred)
            output_ones = torch.ones_like(label_pred)
            label_pred = torch.where((label_pred > 0.5), output_ones, output_zeros)
        label_pred = label_pred.cpu().detach()
        label_pred = np.asarray(label_pred, dtype='int32')

    if type(label_true).__module__ == np.__name__:
        label_true = np.asarray(label_true, dtype='int32')
    else:
        label_true = label_true.cpu().detach()
        label_true = np.asarray(label_true, dtype='int32')

    return label_pred, label_true

# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/utils/metrics.py
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py


def _fast_hist(label_true, label_pred, n_class):
    # label_pred, label_true = preprocessing_accuracy(label_true, label_pred, n_class)
    mask = (label_true >= 0) & (label_true < n_class)

    # print(np.shape(mask))
    # print(np.shape(label_true))
    # print(np.shape(label_pred))

    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def segmentation_scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    label_preds, label_trues = preprocessing_accuracy(label_trues, label_preds, n_class)
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    # iou:
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
    mean_iou = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    # iflat = label_preds.view(-1)
    # tflat = label_trues.view(-1)
    # intersection = (iflat * tflat).sum()
    # union = iflat.sum() + tflat.sum()

    return mean_iou, acc_cls, fwavacc
# ==================================================================================


def f1_score(label_gt, label_pred, n_class):
    # threhold = torch.Tensor([0])
    label_pred, label_gt = preprocessing_accuracy(label_gt, label_pred, n_class)
    #
    if len(label_gt.shape) == 4:
        b, c, h, w = label_gt.shape
        size = b * c
    elif len(label_gt.shape) == 3:
        c, h, w = label_gt.shape
        size = c
    #
    assert len(label_gt) == len(label_pred)
    #
    precision = np.zeros(n_class, dtype=np.float32)
    recall = np.zeros(n_class, dtype=np.float32)
    img_A = np.array(label_gt, dtype=np.float32).flatten()
    img_B = np.array(label_pred, dtype=np.float32).flatten()
    precision[:] = precision_score(img_A, img_B, average=None, labels=range(n_class))
    recall[:] = recall_score(img_A, img_B, average=None, labels=range(n_class))
    f1_metric = 2 * (recall * precision) / (recall + precision + 1e-10)
    # For binary:
    #
    # CM = confusion_matrix(img_A, img_B)
    # #
    # TN = CM[0][0]
    # FN = CM[1][0]
    # TP = CM[1][1]
    # FP = CM[0][1]
    #
    # TN, FP, FN, TP = confusion_matrix(img_A, img_B, labels=[0.0, 1.0]).ravel()
    #
    # TP = np.sum(label_gt[label_gt == 1.0] == label_pred[label_pred == 1.0])
    # TN = np.sum(label_gt[label_gt == 0.0] == label_pred[label_pred == 0.0])
    # FP = np.sum(label_gt[label_gt == 1.0] == label_pred[label_pred == 0.0])
    # FN = np.sum(label_gt[label_gt == 0.0] == label_pred[label_pred == 1.0])
    # For multi-class:
    # FP = CM.sum(axis=0) - np.diag(CM)
    # FN = CM.sum(axis=1) - np.diag(CM)
    # TP = np.diag(CM)
    # TN = CM.sum() - (FP + FN + TP)
    # FPs_Ns = (FP + 1e-8) / ((img_A == float(0.0)).sum() + 1e-8)
    # FNs_Ps = (FN + 1e-8) / ((img_A == float(1.0)).sum() + 1e-8)
    #
    # N = TN + FP
    # P = TP + FN
    #
    # FPs_Ns = (FP + 1e-10) / (Negatives + 1e-10)
    # FNs_Ps = (FN + 1e-10) / (Positives + 1e-10)
    # CM = np.zeros((2, 2), dtype=np.float32)
    #
    return f1_metric.mean(), recall.mean(), precision.mean()

##==========================================================================================

# Hausdorff distance metric:
# adopted from niftynet:
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/niftynet/utilities/util_common.py
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/niftynet/evaluation/pairwise_measures.py


# class MorphologyOps(object):
#     """
#     Class that performs the morphological operations needed to get notably
#     connected component. To be used in the evaluation
#     """
#     #
#     # I did NOT use this function because I don't want to post-processing interfere the segmentation evaluation.
#     #
#     def __init__(self, binary_img, neigh):
#         # assert len(binary_img.shape) == 3, 'currently supports 3d inputs only'
#         #
#         print(len(binary_img.shape))
#         if len(binary_img.shape) == 1:
#             #
#             binary_map_ = torch.cat((binary_img, binary_img, binary_img), dim=1)
#         #
#         self.binary_map = np.asarray(binary_map_, dtype=np.int8)
#         self.neigh = neigh
#
#     def border_map(self):
#         """
#         Creates the border for a 3D image
#         :return:
#         """
#         west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
#         east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
#         north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
#         south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
#         top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
#         bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
#         cumulative = west + east + north + south + top + bottom
#         border = ((cumulative < 6) * self.binary_map) == 1
#         return border
#
#     def foreground_component(self):
#         return ndimage.label(self.binary_map)


# def border_distance(label, output):
#     """
#     This functions determines the map of distance from the borders of the
#     segmentation and the reference and the border maps themselves
#
#     :return: distance_border_ref, distance_border_seg, border_ref,
#         border_seg
#     """
#     # border_ref = MorphologyOps(label, 4).border_map()
#     # border_seg = MorphologyOps(output, 4).border_map()
#     #
#     border_ref = label.cpu()
#     border_ref = np.asarray(border_ref, dtype=np.int8)
#     border_seg = output.cpu()
#     border_seg = np.asarray(border_seg, dtype=np.int8)
#     #
#     oppose_ref = 1 - label
#     oppose_seg = 1 - output
#     # euclidean distance transform
#     distance_ref = ndimage.distance_transform_edt(oppose_ref)
#     distance_seg = ndimage.distance_transform_edt(oppose_seg)
#     distance_border_seg = border_ref * distance_seg
#     distance_border_ref = border_seg * distance_ref
#     return distance_border_ref, distance_border_seg, border_ref, border_seg
#
#
# def getHausdorff(label, output):
#     """
#     This functions calculates the average symmetric distance and the
#     hausdorff distance between a segmentation and a reference image
#     :return: hausdorff distance and average symmetric distance
#     """
#     # label = label.squeeze(0)
#     # output = output.squeeze(1)
#     # label = label.cpu().detach()
#     # output = output.cpu().detach()
#     ref_border_dist, seg_border_dist, ref_border, seg_border = border_distance(label, output)
#     # average_distance = (np.sum(ref_border_dist) + np.sum(
#     #     seg_border_dist)) / (np.sum(label + output))
#     hausdorff_distance = np.max([np.max(ref_border_dist), np.max(seg_border_dist)])
#     seg_values = ref_border_dist[seg_border > 0]
#     ref_values = seg_border_dist[ref_border > 0]
#     if seg_values.size == 0 or ref_values.size == 0:
#         hausdorff95_distance = np.nan
#     else:
#         hausdorff95_distance = np.max([np.percentile(seg_values, 95), np.percentile(ref_values, 95)])
#
#     return hausdorff95_distance, hausdorff_distance


# ==================================

# def perf_measure(y_actual, y_hat):
#
#     TP = 0
#     FP = 0
#     TN = 0
#     FN = 0
#
#     for i in range(len(y_hat)):
#         if y_actual[i]==y_hat[i]==1:
#            TP += 1
#         if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
#            FP += 1
#         if y_actual[i]==y_hat[i]==0:
#            TN += 1
#         if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
#            FN += 1
#
#     return TP, FP, TN, FN


# =============================
# reference :http://loli.github.io/medpy/_modules/medpy/metric/binary.html


def __surface_distances(result, reference, class_no, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result, reference = preprocessing_accuracy(reference, result, class_no)
    # reference = reference.cpu().detach().numpy()
    # result = result.cpu().detach().numpy()
    #
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def hd95(result, reference, class_no, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, class_no, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, class_no, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    #
    hd95_mean = np.nanmean(hd95)
    return hd95_mean