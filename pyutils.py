#!/usr/bin/env python
# coding=utf-8
import os
import torch
import random
import importlib
import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def to_device(d, device):
    if isinstance(d, dict):
        return {k: to_device(v, device) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_device(v, device) for v in d]
    else:
        assert(isinstance(d, torch.Tensor) or isinstance(d, np.ndarray))
        return d.to(device)


def to_numpy(d):
    if isinstance(d, torch.Tensor):
        return d.detach().cpu().numpy()
    if isinstance(d, dict):
        return {k: to_numpy(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_numpy(v) for v in d]
    else:
        return d


def load_module(module_name, class_name=None):
    module = importlib.import_module(module_name)
    if class_name is not None:
        return getattr(module, class_name)
    else:
        return module


def update_config_from_args(config, args):
    for k, v in args.__dict__.items():
        config[k] = v


