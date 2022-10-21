import torch
from pathlib import Path

import random
import numpy as np
import torch.backends.cudnn as cudnn

# model:
from Models2D import Unet, UnetBPL

# data:
from libs.Dataloader import getData

# track the training
from tensorboardX import SummaryWriter


def reproducibility(args):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def network_intialisation(args):
    if args.unlabelled == 0:
        # supervised learning:
        model = Unet(in_ch=args.input_dim,
                     width=args.width,
                     depth=args.depth,
                     classes=args.output_dim,
                     norm='in',
                     side_output=False)

        model_name = 'Unet_l' + str(args.lr) + \
                       '_b' + str(args.batch) + \
                       '_w' + str(args.width) + \
                       '_d' + str(args.depth) + \
                       '_i' + str(args.iterations) + \
                       '_l2_' + str(args.l2) + \
                       '_c_' + str(args.contrast) + \
                       '_n_' + str(args.norm) + \
                       '_t' + str(args.temp)

    else:
        # supervised learning plus pseudo labels:
        model = UnetBPL(in_ch=args.input_dim,
                        width=args.width,
                        depth=args.depth,
                        out_ch=args.output_dim,
                        norm='in',
                        ratio=8,
                        detach=args.detach)

        model_name = 'BPUnet_l' + str(args.lr) + \
                       '_b' + str(args.batch) + \
                       '_u' + str(args.unlabelled) + \
                       '_w' + str(args.width) + \
                       '_d' + str(args.depth) + \
                       '_i' + str(args.iterations) + \
                       '_l2_' + str(args.l2) + \
                       '_c_' + str(args.contrast) + \
                       '_n_' + str(args.norm) + \
                       '_t' + str(args.temp) + \
                       '_de_' + str(args.detach) + \
                       '_mu' + str(args.mu) + \
                       '_sig' + str(args.sigma) + \
                       '_a' + str(args.alpha) + \
                       '_w' + str(args.warmup)

    return model, model_name


def make_saving_directories(model_name, args):
    save_model_name = model_name
    saved_information_path = '../Results/' + args.log_tag
    Path(saved_information_path).mkdir(parents=True, exist_ok=True)
    saved_log_path = saved_information_path + '/Logs'
    Path(saved_log_path).mkdir(parents=True, exist_ok=True)
    saved_model_path = saved_information_path + '/' + save_model_name + '/trained_models'
    Path(saved_model_path).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(saved_log_path + '/Log_' + save_model_name)
    return writer, saved_model_path


def get_iterators(args):
    data_loaders = getData(data_directory=args.data,
                           train_batchsize=args.batch,
                           norm=args.norm,
                           zoom_aug=args.zoom,
                           sampling_weight=args.sampling,
                           contrast_aug=args.contrast,
                           lung_window=args.lung_window,
                           unlabelled=args.unlabelled)

    return data_loaders


def get_data_dict(dataloader, iterator):
    try:
        data_dict, data_name = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        data_dict, data_name = next(iterator)
    del data_name
    return data_dict


def ramp_up(weight, ratio, step, total_steps, starting=50):
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
    ramp_up_length = int(ratio*total_steps)
    if step > starting:
        current_weight = weight * (step-starting) / ramp_up_length
    else:
        current_weight = 0.0
    return min(current_weight, weight)








