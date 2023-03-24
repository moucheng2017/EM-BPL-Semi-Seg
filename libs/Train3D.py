import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.Loss import SoftDiceLoss, kld_loss
from libs.Metrics import segmentation_scores
from libs.Helpers import np2tensor_all, check_dim, check_inputs, get_img, model_forward


def calculate_sup_loss(lbl,
                       raw_output,
                       temp):
    # outputs_dict,
    prob_output = torch.sigmoid(raw_output / temp)
    pseudo_label = (prob_output > 0.5).float()
    # _, pseudo_label = torch.max(raw_output, dim=1)
    # prob_output = torch.softmax(raw_output / temp, dim=1)

    if len(lbl.size()) == 5:
        (b, c, d, h, w) = lbl.size()
        vol = b*d*h*w*c
    elif len(lbl.size()) == 4:
        (b, d, h, w) = lbl.size()
        vol = b*d*h*w
    else:
        raise NotImplementedError

    if lbl.sum() > 0.0005*vol: # at least 0.5% foreground
        loss_sup = 0.5*SoftDiceLoss()(prob_output, lbl) + 0.5*F.binary_cross_entropy_with_logits(raw_output, lbl)
    else:
        loss_sup = torch.zeros(1).cuda()

    if prob_output.size()[1] == 1:
        classes = 2
    else:
        classes = prob_output.size()[1]
    train_mean_iu_ = segmentation_scores(lbl.squeeze(), pseudo_label.squeeze(), classes)

    return {'loss': loss_sup,
            'train iou': train_mean_iu_,
            'prob': prob_output.mean()}


def train_semi(labelled_img,
               labelled_label,
               model,
               unlabelled_img,
               t=2.0,
               pri_mu=0.8,
               pri_std=0.1,
               flag_post_mu=0,
               flag_post_std=0,
               flag_pri_mu=0,
               flag_pri_std=0
               ):

    # convert data from numpy to tensor:
    inputs = np2tensor_all(**{'img_l': labelled_img,
                              'lbl': labelled_label,
                              'img_u': unlabelled_img})

    # concatenate labelled and unlabelled for ssl otherwise just use labelled img
    train_img = get_img(**inputs)

    # forward pass:
    img = train_img['train img']
    if len(img.size()) == 4: # for normal 1D channel data of 3d volumetric
        img = img.unsqueeze(1)
    elif len(img.size()) == 5: # for multi-channel data such as BRATS
        assert img.size()[1] > 1
    else:
        raise NotImplementedError

    outputs_dict = model_forward(model, img)
    b_l = train_img['batch labelled']
    b_u = train_img['batch unlabelled']

    # get output for the labelled part:
    raw_output = outputs_dict['segmentation']
    raw_output_l, raw_output_u = torch.split(raw_output, [b_l, b_u], dim=0)

    # supervised loss:
    sup_loss = calculate_sup_loss(raw_output=raw_output_l,
                                  lbl=inputs['lbl'].unsqueeze(1),
                                  temp=t)

    # calculate the kl and get the learnt threshold:
    kl_loss = calculate_kl_loss(outputs_dict=outputs_dict,
                                b_l=train_img['batch labelled'],
                                b_u=train_img['batch unlabelled'],
                                pri_mu=pri_mu,
                                pri_std=pri_std,
                                flag_post_mu=flag_post_mu,
                                flag_post_std=flag_post_std,
                                flag_pri_mu=flag_pri_mu,
                                flag_pri_std=flag_pri_std
                                )

    # pseudo label loss:
    pseudo_loss = calculate_pseudo_loss(raw_output=raw_output_u,
                                        threshold=kl_loss['threshold'],
                                        temp=t
                                        )

    return {'supervised losses': sup_loss,
            'pseudo losses': pseudo_loss,
            'kl losses': kl_loss}


def calculate_pseudo_loss(raw_output,
                          threshold,
                          temp):

    prob_output = torch.sigmoid(raw_output / temp)
    pseudo_labels = (prob_output > threshold).float()

    # for unlabelled parts:
    if len(pseudo_labels.size()) == 5:
        (b, c, d, h, w) = pseudo_labels.size()
        vol = b*d*h*w*c
    elif len(pseudo_labels.size()) == 4:
        (b, d, h, w) = pseudo_labels.size()
        vol = b*d*h*w
    else:
        raise NotImplementedError

    if pseudo_labels.sum() > 0.0005*vol:
        loss_unsup = 0.5*SoftDiceLoss()(prob_output, pseudo_labels)
        loss_unsup += 0.5*F.binary_cross_entropy_with_logits(raw_output, pseudo_labels)
    else:
        loss_unsup = torch.zeros(1).cuda()

    return {'loss': loss_unsup,
            'prob': prob_output.mean()}


def train_sup(labelled_img,
              labelled_label,
              model,
              t=2.0):

    inputs = np2tensor_all(**{'img_l': labelled_img, 'lbl': labelled_label})
    train_img = get_img(**inputs)

    img = train_img.get('train img')
    if len(img.size()) == 4: # for normal 1D channel data of 3d volumetric
        img = img.unsqueeze(1)
    elif len(img.size()) == 5: # for multi-channel data such as BRATS
        assert img.size()[1] > 1
    else:
        raise NotImplementedError

    outputs_dict = model_forward(model, img)
    raw_output = outputs_dict['segmentation']
    sup_loss = calculate_sup_loss(raw_output=raw_output,
                                  lbl=inputs['lbl'].unsqueeze(1),
                                  temp=t)

    return {'supervised losses': sup_loss}


def calculate_kl_loss(outputs_dict,
                      b_u,
                      b_l,
                      pri_mu,
                      pri_std,
                      flag_post_mu,
                      flag_post_std,
                      flag_pri_mu,
                      flag_pri_std
                      ):

    assert b_u > 0
    posterior_mu = outputs_dict['mu']
    posterior_logvar = outputs_dict['logvar']
    raw_output = outputs_dict['segmentation']

    loss, confidence_threshold_learnt = kld_loss(raw_output=raw_output,
                                                 mu1=posterior_mu,
                                                 logvar1=posterior_logvar,
                                                 mu2=pri_mu,
                                                 std2=pri_std,
                                                 flag_mu1=flag_post_mu,
                                                 flag_std1=flag_post_std,
                                                 flag_mu2=flag_pri_mu,
                                                 flag_std2=flag_pri_std
                                                 )

    # confidence_threshold_learnt_l, confidence_threshold_learnt_u = torch.split(confidence_threshold_learnt, [b_l, b_u], dim=0)

    return {'loss': loss.mean(),
            'threshold': confidence_threshold_learnt}


# confidence mask:
    # mask = prob_output.ge(0.95)
    # mask the labels:
    # prob_output_selected = prob_output*mask
    # raw_output_selected = raw_output*mask
    # pseudo_label_selected = pseudo_label*mask
    # print(pseudo_label_selected.size())
    # print(raw_output_selected.size())


# def calculate_pseudo_loss(outputs_dict,
#                           b_u,
#                           b_l,
#                           temp,
#                           lbl,
#                           cutout_aug=0,
#                           conf_threshold='bayesian'):
#
#     assert b_u > 0
#     predictions_all = outputs_dict.get('segmentation')
#
#     # Monte Carlo sampling of confidence threshold:
#     if conf_threshold == 'bayesian':
#         threshold = outputs_dict.get('learnt_threshold')
#     else:
#         threshold = 0.5
#
#     predictions_l, predictions_u = torch.split(predictions_all, [b_l, b_u], dim=0)
#     threshold_l, threshold_u = torch.split(threshold, [b_l, b_u], dim=0)
#     prob_output_u = torch.softmax(predictions_u / temp, dim=1)
#     pseudo_label_u = (prob_output_u >= threshold_u).float()
#     prob_output_l = torch.softmax(predictions_l / temp, dim=1)
#     pseudo_label_l = (prob_output_l >= threshold_l).float()
#
#     # if cutout_aug == 1:
#     #     prob_output_u, pseudo_label_u = randomcutout(prob_output_u, pseudo_label_u)
#
#     mask = torch.zeros_like(lbl)
#
#     if torch.sum(pseudo_label_u) > 10:
#         if len(prob_output_u.size()) == 3:
#             # this is binary segmentation
#             loss = SoftDiceLoss()(prob_output_u, pseudo_label_u) + nn.BCELoss(reduction='mean')(prob_output_u.squeeze() + 1e-10, pseudo_label_u.squeeze() + 1e-10)
#             loss += 0.5*SoftDiceLoss()(prob_output_l, pseudo_label_l) + nn.BCELoss(reduction='mean')(prob_output_l.squeeze() + 1e-10, pseudo_label_l.squeeze() + 1e-10)
#             return {'loss': loss.mean()}
#
#         elif len(prob_output_u.size()) == 4:
#             if prob_output_u.size()[1] == 1:
#                 # this is also binary segmentation
#                 loss = SoftDiceLoss()(prob_output_u, pseudo_label_u) + nn.BCELoss(reduction='mean')(prob_output_u.squeeze() + 1e-10, pseudo_label_u.squeeze() + 1e-10)
#                 loss += 0.5*SoftDiceLoss()(prob_output_l, pseudo_label_l) + nn.BCELoss(reduction='mean')(prob_output_l.squeeze() + 1e-10, pseudo_label_l.squeeze() + 1e-10)
#                 return {'loss': loss.mean()}
#
#             else:
#                 # this is multi class segmentation
#                 pseudo_label_u = multi_class_label_processing(pseudo_label_u, prob_output_u.size()[1])  # convert single channel multi integer class label to multi channel binary label
#                 loss = torch.tensor(0).to('cuda')
#                 effective_classes = 0
#                 for i in range(prob_output_u.size()[1]):  # multiple
#                     if torch.sum(pseudo_label_u[:, i, :, :]) > 10.0:
#                         # If the channel is not empty, we learn it otherwise we ignore that channel because sometimes we do learn some very weird stuff
#                         # It is necessary to use this condition because some labels do not necessarily contain all of the classes in one image.
#                         effective_classes += 1
#                         loss += SoftDiceLoss()(prob_output_u[:, i, :, :], pseudo_label_u[:, i, :, :]).mean() + nn.BCELoss(reduction='mean')(prob_output_u[:, i, :, :].squeeze() + 1e-10, pseudo_label_u[:, i, :, :].squeeze() + 1e-10).mean()
#                         loss += 0.5*SoftDiceLoss()(prob_output_l[:, i, :, :], pseudo_label_l[:, i, :, :]).mean() + nn.BCELoss(reduction='mean')(prob_output_l[:, i, :, :].squeeze() + 1e-10, pseudo_label_l[:, i, :, :].squeeze() + 1e-10).mean()
#                 loss = loss / effective_classes
#                 return {'loss': loss.mean()}
#
#     else:
#         return {'loss': torch.tensor(0.0).to('cuda').mean()}






