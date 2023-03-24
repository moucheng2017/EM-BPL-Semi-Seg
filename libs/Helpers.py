import os
import torch
from pathlib import Path
import glob
import random
import numpy as np
import torch.backends.cudnn as cudnn
import nibabel as nib

from models.Models3D import Unet3D, UnetBPL3D
from libs.Dataloader3D import getData3D

# track the training
from tensorboardX import SummaryWriter


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


def reproducibility(args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def network_intialisation(args):
    if args.train.batch_u == 0:
        # supervised learning:
        model = Unet3D(in_ch=args.model.input_dim,
                       width=args.model.width,
                       depth=args.model.depth,
                       classes=args.model.output_dim,
                       side_output=False)

        model_name = 'Unet3D_l_' + str(args.train.lr) + \
                     '_b' + str(args.train.batch) + \
                     '_w' + str(args.model.width) + \
                     '_d' + str(args.model.depth) + \
                     '_i' + str(args.train.iterations) + \
                     '_cd' + str(args.train.new_size_d) + \
                     '_ch' + str(args.train.new_size_h) + \
                     '_cw' + str(args.train.new_size_w)

    else:
        model = UnetBPL3D(in_ch=args.model.input_dim,
                          width=args.model.width,
                          depth=args.model.depth,
                          out_ch=args.model.output_dim,
                          )

        model_name = 'BPL3D_l_' + str(args.train.lr) + \
                     '_b' + str(args.train.batch) + \
                     '_w' + str(args.model.width) + \
                     '_d' + str(args.model.depth) + \
                     '_i' + str(args.train.iterations) + \
                     '_u' + str(args.train.batch_u) + \
                     '_m2' + str(args.train.pri_mu) + \
                     '_std2' + str(args.train.pri_std) + \
                     '_fm1' + str(args.train.flag_post_mu) + \
                     '_fstd1' + str(args.train.flag_post_std) + \
                     '_fm2' + str(args.train.flag_pri_mu) + \
                     '_fstd2' + str(args.train.flag_pri_std) + \
                     '_cd' + str(args.train.new_size_d) + \
                     '_ch' + str(args.train.new_size_h) + \
                     '_cw' + str(args.train.new_size_w)

    return model, model_name


def make_saving_directories(model_name, args):
    save_model_name = model_name
    dataset_name = os.path.basename(os.path.normpath(args.dataset.data_dir))
    saved_information_path = '../../Results_' + dataset_name + '/' + args.logger.tag
    Path(saved_information_path).mkdir(parents=True, exist_ok=True)
    saved_log_path = saved_information_path + '/Logs'
    Path(saved_log_path).mkdir(parents=True, exist_ok=True)
    saved_model_path = saved_information_path + '/' + save_model_name + '/trained_models'
    Path(saved_model_path).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)
    return writer, saved_model_path


def get_iterators(args):

    data_loaders = getData3D(data_directory=args.dataset.data_dir,
                             train_batchsize=args.train.batch,
                             crop_aug=args.train.crop_aug,
                             num_workers=args.dataset.num_workers,
                             transpose_dim=args.train.transpose_dim,
                             gaussian_aug=args.train.gaussian,
                             data_format=args.dataset.data_format,
                             contrast_aug=args.train.contrast,
                             unlabelled=args.train.batch_u,
                             output_shape=(args.train.new_size_d, args.train.new_size_h, args.train.new_size_w))

    return data_loaders


def get_data_dict(dataloader, iterator):

    try:
        data_dict, data_name = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        data_dict, data_name = next(iterator)

    del data_name
    return data_dict


def ramp_up(weight,
            ratio,
            step,
            total_steps,
            starting):
    '''
    Args:
        weight: final target weight value
        ratio: ratio between the length of ramping up and the total steps
        step: current step
        total_steps: total steps
        starting: starting step for ramping up
    Returns:
        current weight value
    '''
    # For the 1st 50 steps, the weighting is zero
    # For the ramp-up stage from starting through the length of ramping up, we linearly gradually ramp up the weight
    starting = starting*total_steps
    ramp_up_length = ratio*total_steps

    if step < starting:
        return 0.0
    elif step < (ramp_up_length+starting):
        current_weight = weight * (step-starting) / ramp_up_length
        return min(current_weight, weight)
    else:
        return weight








