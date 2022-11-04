import contextlib
import sys
import logging
import math
import time
import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.special import logsumexp
from ema import EMA
from model import *
from utils import *
from diffusion_utils import *

plt.style.use("ggplot")

# TODO: Add logger


class Diffusion(object):
    def __init__(self, args, config, device=None):
        """
        config is the yml file
        """
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.num_timesteps = config.diffusion.timesteps
        self.dataset_object = None

        # Model related expressions setup
        betas: torch.Tensor = make_beta_schedule(
            scheduler=config.diffusion.beta_schedule,
            num_timesteps=self.num_timesteps,
            start=config.different.beta_start,
            end=config.diffusion.beta_end,
        )

        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas: torch.Tensor = 1.0 - betas
        self.alphas: torch.Tensor = alphas
        self.one_minux_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)

        if config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0

        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
            torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance

        # TODO: add other conditioning signal such as OLS
        # Initial prediction model as guided condition
        # Forward process
        self.cond_pred_model = DeterministicFeedForwardNeuralNetwork(
            dim_in=config.model.x_dim,
            dim_out=config.model.y_dim,
            hid_layers=config.dissusion.nonlinear_guidance.hid_layers,
            use_batchnorm=config.diffusion.nonlinear_guidance.use_batchnorm,
            negative_slope=config.diffusion.nonlinear_guidance.negative_slope,
            dropout_rate=config.diffusion.nonlinear_guidance.dropout_rate,
        ).to(self.device)
        self.aux_cost_function = nn.MSELoss()

    def compute_guiding_prediction(self, x, method="NN"):
        return self.cond_pred_model(x) if method == "NN" else None

    def evaluate_guidance_model(self, dataset_object, dataset_loader):
        """
        Evaluate guidance model by reporting train or test set unnoramlized y RMSe.
        aka Root mean square deviation
        """
        y_se_list = []
        for xy_0 in dataset_loader:
            xy_0 = xy_0.to(self.device)
            x_batch = xy_0[:, : -self.config.model.y_dim]
            y_batch = xy_0[:, -self.config.model.y_dim :]
            y_batch_pred_mean = (
                self.compute_guiding_prediction(
                    x_batch, method=self.config.diffusion.conditioning_signal
                )
                .cpu()
                .detach()
                .numpy()
            )
            y_batch = y_batch.cpu().detach().numpy()
            if dataset_object.normalize_y:
                y_batch = dataset_object.scaler_y.inverse_transform(y_batch).astype(
                    np.float32
                )
                y_batch_pred_mean = dataset_object.scaler_y.inverse_transform(
                    y_batch_pred_mean
                ).astype(np.float32)
                y_se = (y_batch_pred_mean - y_batch) ** 2
                if len(y_se_list) == 0:
                    y_se_list = y_se
                else:
                    y_se_list = np.concatenate([y_se_list, y_se], axis=0)
        return np.sqrt(np.mean(y_se_list))

    def nonlinear_guidance_model_train_step(self, x_batch, y_batch, aux_optimizer):
        """
        One optimization step of the non-linear guidance model that predits y_0_hat.
        """
        y_batch_pred = self.cond_pred_model(x_batch)
        aux_cost = self.aux_cost_function(y_batch_pred, y_batch)  # nn.MSELoss()
        # update non-linear guidance model
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        return aux_cost.cpu().item()

    def nonlinear_guidance_model_train_loop_per_epoch(
        self, train_batch_loader, aux_optimizer, epoch
    ):
        for xy_0 in train_batch_loader:
            xy_0 = xy_0.to(self.device)
            x_batch = xy_0[:, : -self.config.model.y_dim]
            y_batch = xy_0[:, -self.config.model.y_dim :]
            aux_loss = self.nonlinear_guidance_model_train_step(
                x_batch, y_batch, aux_optimizer
            )
        if epoch % self.config.diffusion.nonlinear_guidance.logging_interval == 0:
            logging.info(
                f"epoch: {epoch}, non-linear guidance model pre-training loss: {aux_loss}"
            )
        return None

    def obtain_true_and_pred_y_t(self, cur_t, y_seq, y_T_mean, y_0):
        """
        evaluate: return $y_t_p_sample, y_t_true$
        """
        y_t_p_sample = y_seq[self.num_timesteps - cur_t].detach().cpu()
        y_t_true = (
            q_sample(
                y_0,
                y_T_mean,
                self.alphas_bar_sqrt,
                self.one_minus_alphas_bar_sqrt,
                torch.tensor([cur_t - 1]),
            )
            .detach()
            .cpu()
        )
        return y_t_p_sample, y_t_true

    def compute_unnorm_y(self, cur_y, testing):
        if testing:
            y_mean = (
                cur_y.cpu()
                .reqshape(-1, self.config.testing.n_z_samplse)
                .mean(1)
                .reshape(-1, 1)
            )
        else:
            y_mean = cur_y.cpu()
        return (
            self.dataset_object.scaler_y.inverse_transform(y_mean)  # From sklearn
            if self.config.data.normalize_y
            else y_mean
        )

    def train(self):
        args = self.args
        config = self.config

        logging.info("Test set info:")
        test_set_object, test_set = get_dataset(args, config, test_set=True)
        test_loader = data.DataLoader(
            test_set,
            batch_size=config.testing.batch_size,
            num_workers=config.data.num_workers,
        )

        # obtain training set
        logging.info("training set info:")
        dataset_object, dataset = get_dataset(args, config, test_set=False)
        self.dataset_object = dataset_object
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        # setup model
        model = ConditionalGuidedModel(config)
        model = model.to(self.device)

        # evaluate f_phi(x) on both training and test set
        logging.info("\nBefore pre-training:")
        # self.evaluate_guidance_model_on_both_train_and_test_set(
        #     dataset_object, train_loader, test_set_object, test_loader
        # )

        # setup optimizer
        optimizer = get_optimizer(self.config.optim, model.parameters())

        # apply an auxiliary optimizer for the NN guidance model taht predict y_0_hat
        aux_optimizer = get_optimizer(
            self.config.aux_optim, self.conf_pred_model.parameters()
        )
