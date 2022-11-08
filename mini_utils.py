import random
import logging
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch import nn
from mini_data_loader import *

# from data_loader import *


def set_random_seed(seed):
    print(f"\n* Set seed {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        new_value = dict2namespace(value) if isinstance(value, dict) else value
        setattr(namespace, key, new_value)
    return namespace


def get_dataset(args, config, test_set=False):
    data_object = None
    if config.data.dataset != "linear_regression":
        raise NotImplementedError(
            "Toy dataset options: swiss_roll, moons, 8gaussians and 25gaussians; regression data: UCI."
        )
    data_object = LinearDatasetWithOneX(
        a=config.data.a,
        b=config.data.b,
        n_samples=config.data.dataset_size,
        seed=args.seed,
        x_dict=vars(config.data.x_dict),
        noise_dict=vars(config.data.noise_dict),
        normalize_x=config.data.normalize_x,
        normalize_y=config.data.normalize_y,
    )
    data_object.create_train_test_dataset(train_ratio=config.data.train_ratio)
    data = data_object.test_dataset if test_set else data_object.train_dataset
    return data_object, data


def get_optimizer(config_optim, parameters):
    if config_optim.optimizer == "Adam":
        return optim.Adam(
            parameters,
            lr=config_optim.lr,
            weight_decay=config_optim.weight_decay,
            betas=(config_optim.beta1, 0.999),
            amsgrad=config_optim.amsgrad,
            eps=config_optim.eps,
        )
    elif config_optim.optimizer == "RMSProp":
        return optim.RMSprop(
            parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay
        )
    elif config_optim.optimizer == "SGD":
        return optim.SGD(parameters, lr=config_optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(f"Optimizer {config_optim.optimizer} not understood.")
