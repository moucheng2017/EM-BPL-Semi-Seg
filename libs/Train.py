import torch
import torch.nn as nn
from Loss import SoftDiceLoss, kld_loss
from Metrics import segmentation_scores
from libs.Augmentations import randomcutout
from libs.LabelEncoding import multi_class_label_processing


def check_dim(input_tensor):
    '''
    Args:
        input_tensor:
    Returns:
    '''
    if len(input_tensor.size()) < 4:
        return input_tensor.unsqueeze(1)
    else:
        return input_tensor


def check_inputs(**kwargs):
    outputs = {}
    for key, val in kwargs.items():
        # check the dimension for each input
        outputs[key] = check_dim(val)
    return outputs


def np2tensor_all(**kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outputs = {}
    for key, val in kwargs.items():
        outputs[key] = val.to(device=device, dtype=torch.float32)
    outputs = check_inputs(**outputs)
    return outputs


def get_img(**inputs):
    img_l = inputs.get('img_l')
    img_u = inputs.get('img_u')
    if img_u is not None:
        img = torch.cat((img_l, img_u), dim=0)
        b_l = img_l.size()[0]
        b_u = img_u.size()[0]
        del img_l
        del img_u
        return {'train img': img,
                'batch labelled': b_l,
                'batch unlabelled': b_u}
    else:
        return {'train img': img_l}


def model_forward(model, img):
    return model(img)


def calculate_sup_loss(lbl,
                       outputs_dict,
                       b_u,
                       b_l,
                       temp,
                       cutout_aug):

    if torch.sum(lbl) > 10: # check whether there are enough foreground pixels
        prob_output = outputs_dict.get('segmentation') # get segmentation map
        if b_u > 0: # check if unlabelled data is included
            prob_output, _ = torch.split(prob_output, [b_l, b_u], dim=0) # if both labelled and unlabelled, split the data and use the labelled

        prob_output = torch.sigmoid(prob_output / temp) # apply element-wise sigmoid

        if cutout_aug is True: # apply cutout augmentation
            prob_output, lbl = randomcutout(prob_output, lbl)

        # channel-wise loss (this is for multi-channel sigmoid function as well):
        if len(prob_output.size()) == 3:
            # this is binary segmentation
            loss = SoftDiceLoss()(prob_output, lbl) + nn.BCELoss(reduction='mean')(prob_output.squeeze() + 1e-10, lbl.squeeze() + 1e-10)
            class_outputs = (prob_output > 0.95).float()
            train_mean_iu_ = segmentation_scores(lbl, class_outputs, 2)
            train_mean_iu_ = sum(train_mean_iu_) / len(train_mean_iu_)
            return {'loss': loss.mean(),
                    'train iou': train_mean_iu_}

        elif len(prob_output.size()) == 4:
            if prob_output.size()[1] == 1:
                # this is also binary segmentation
                loss = SoftDiceLoss()(prob_output, lbl) + nn.BCELoss(reduction='mean')(prob_output.squeeze() + 1e-10, lbl.squeeze() + 1e-10)
                class_outputs = (prob_output > 0.95).float()
                train_mean_iu_ = segmentation_scores(lbl, class_outputs, 2)
                train_mean_iu_ = sum(train_mean_iu_) / len(train_mean_iu_)
                return {'loss': loss.mean(),
                        'train iou': train_mean_iu_}

            else:
                # this is multi class segmentation
                lbl = multi_class_label_processing(lbl, prob_output.size()[1]) # convert single channel multi integer class label to multi channel binary label
                loss = 0
                train_mean_iu_ = 0
                effective_classes = 0
                for i in range(prob_output.size()[1]): # multiple
                    if torch.sum(lbl[:, i, :, :]) > 1.0:
                        # If the channel is not empty, we learn it otherwise we ignore that channel because sometimes we do learn some very weird stuff
                        # It is necessary to use this condition because some labels do not necessarily contain all of the classes in one image.
                        effective_classes += 1
                        loss += SoftDiceLoss()(prob_output[:, i, :, :], lbl[:, i, :, :]).mean() + nn.BCELoss(reduction='mean')(prob_output[:, i, :, :].squeeze() + 1e-10, lbl[:, i, :, :].squeeze() + 1e-10).mean()
                        class_outputs = (prob_output[:, i, :, :] > 0.95).float()
                        train_mean_iu_list = segmentation_scores(lbl[:, i, :, :], class_outputs, 2)
                        train_mean_iu_ += sum(train_mean_iu_list) / len(train_mean_iu_list)

                loss = loss / effective_classes
                train_mean_iu_ = train_mean_iu_ / effective_classes
                return {'loss': loss,
                        'train iou': train_mean_iu_}

        else:
            print('the output is probably 3D and we do not support it yet')

    else:
        loss = 0.0
        train_mean_iu_ = 0.0
        return {'loss': loss,
                'train iou': train_mean_iu_}


def calculate_kl_loss(outputs_dict,
                      b_u,
                      b_l,
                      prior_u,
                      prior_var):

    assert b_u > 0
    posterior_mu = outputs_dict.get('mu')
    posterior_logvar = outputs_dict.get('logvar')

    confidence_threshold_learnt = outputs_dict.get('learnt_threshold')

    loss = kld_loss(posterior_mu, posterior_logvar, prior_u, prior_var)

    threshold_learnt_l, threshold_learnt_u = torch.split(confidence_threshold_learnt, [b_l, b_u], dim=0)

    return {'loss': loss.mean(),
            'threshold labelled': threshold_learnt_l,
            'threshold unlabelled': threshold_learnt_u}


def calculate_pseudo_loss(outputs_dict,
                          b_u,
                          b_l,
                          temp,
                          cutout_aug,
                          conf_threshold='bayesian'):

    assert b_u > 0
    predictions_all = outputs_dict.get('segmentation')

    # Monte Carlo sampling of confidence threshold:
    if conf_threshold == 'bayesian':
        threshold = outputs_dict.get('learnt_threshold')
    else:
        threshold = 0.5

    _, predictions_u = torch.split(predictions_all, [b_l, b_u], dim=0)
    _, threshold_u = torch.split(threshold, [b_l, b_u], dim=0)
    prob_output_u = torch.sigmoid(predictions_u / temp)
    pseudo_label_u = (prob_output_u > threshold_u).float()

    if cutout_aug is True:
        prob_output_u, pseudo_label_u = randomcutout(prob_output_u, pseudo_label_u)

    if len(prob_output_u.size()) == 3:
        # this is binary segmentation
        loss = SoftDiceLoss()(prob_output_u, pseudo_label_u) + nn.BCELoss(reduction='mean')(prob_output_u.squeeze() + 1e-10, pseudo_label_u.squeeze() + 1e-10)
        return {'loss': loss.mean()}

    elif len(prob_output_u.size()) == 4:
        if prob_output_u.size()[1] == 1:
            # this is also binary segmentation
            loss = SoftDiceLoss()(prob_output_u, pseudo_label_u) + nn.BCELoss(reduction='mean')(prob_output_u.squeeze() + 1e-10, pseudo_label_u.squeeze() + 1e-10)
            return {'loss': loss.mean()}

        else:
            # this is multi class segmentation
            pseudo_label_u = multi_class_label_processing(pseudo_label_u, prob_output_u.size()[1])  # convert single channel multi integer class label to multi channel binary label
            loss = 0
            effective_classes = 0
            for i in range(prob_output_u.size()[1]):  # multiple
                if torch.sum(pseudo_label_u[:, i, :, :]) > 1.0:
                    # If the channel is not empty, we learn it otherwise we ignore that channel because sometimes we do learn some very weird stuff
                    # It is necessary to use this condition because some labels do not necessarily contain all of the classes in one image.
                    effective_classes += 1
                    loss += SoftDiceLoss()(prob_output_u[:, i, :, :], pseudo_label_u[:, i, :, :]).mean() + nn.BCELoss(reduction='mean')(prob_output_u[:, i, :, :].squeeze() + 1e-10, pseudo_label_u[:, i, :, :].squeeze() + 1e-10).mean()
            loss = loss / effective_classes
            return {'loss': loss}

    else:
        print('the output is probably 3D and we do not support it yet')


def train_sup(labelled_img,
              labelled_label,
              model,
              t=2.0,
              augmentation_cutout=True):

    inputs = np2tensor_all(**{'img_l':labelled_img, 'lbl':labelled_label})
    train_img = get_img(**inputs)
    outputs_dict = model_forward(model, train_img.get('train img'))
    sup_loss = calculate_sup_loss(outputs_dict=outputs_dict,
                                  lbl=inputs.get('lbl'),
                                  temp=t,
                                  b_l=train_img.get('batch labelled'),
                                  b_u=train_img.get('batch unlabelled'),
                                  cutout_aug=augmentation_cutout)
    return {'supervised loss': sup_loss}


def train_semi(labelled_img,
               labelled_label,
               model,
               unlabelled_img,
               t=2.0,
               prior_mu=0.6,
               prior_logsigma=0.1,
               augmentation_cutout=True):

    # convert data from numpy to tensor:
    inputs = np2tensor_all(**{'img_l': labelled_img,
                              'lbl': labelled_label,
                              'img_u': unlabelled_img})

    # concatenate labelled and unlabelled for ssl otherwise just use labelled img
    train_img = get_img(**inputs)

    # forward pass:
    outputs_dict = model_forward(model, train_img.get('train img'))

    # supervised loss:
    sup_loss = calculate_sup_loss(outputs_dict=outputs_dict,
                                  lbl=inputs.get('lbl'),
                                  temp=t,
                                  b_l=train_img.get('batch labelled'),
                                  b_u=train_img.get('batch unlabelled'),
                                  cutout_aug=augmentation_cutout)

    # calculate the kl and get the learnt threshold:
    kl_loss = calculate_kl_loss(outputs_dict=outputs_dict,
                                b_l=train_img.get('batch labelled'),
                                b_u=train_img.get('batch unlabelled'),
                                prior_u=prior_mu,
                                prior_var=prior_logsigma)

    # pseudo label loss:
    pseudo_loss = calculate_pseudo_loss(outputs_dict=outputs_dict,
                                        b_l=train_img.get('batch labelled'),
                                        b_u=train_img.get('batch unlabelled'),
                                        temp=t,
                                        cutout_aug=augmentation_cutout
                                        )

    return {'supervised loss': sup_loss,
            'pseudo loss': pseudo_loss,
            'kl loss': kl_loss}

