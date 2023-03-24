import argparse
import os
import torch

import numpy as np
import torch
import random

import re
import yaml

import shutil
import warnings

from datetime import datetime


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your configs file(xxx.yaml)!")


def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configs-file', required=True, type=str, help="xxx.yaml")
    args = parser.parse_args()

    with open(args.configs_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    set_deterministic(args.seed)

    # vars(args)['aug_kwargs'] = {
    #     'name': args.model.name
    # }

    vars(args)['dataset_kwargs'] = {
        'dataset': args.dataset.name,
        'data_dir': args.dataset.data_dir,
        'data_format': args.dataset.data_format
    }

    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    vars(args)['train_kwargs'] = {
        'lr': args.train.lr,
        'iterations': args.train.iterations,
        'batch': args.train.batch,
        'weight_decay': args.train.optimizer.weight_decay
    }

    # vars(args)['eval_kwargs'] = {
    #     'batch': args.checkpoint.checkpoint_path
    # }

    vars(args)['checkpoint_kwargs'] = {
        'resume': args.checkpoint.resume,
        'checkpoint_path': args.checkpoint.checkpoint_path
    }

    return args
