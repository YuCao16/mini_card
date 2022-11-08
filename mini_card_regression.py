import contextlib
import logging
import time

# import sys, math, gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

# from scipy.special import logsumexp
from ema import EMA
from mini_model import *
from mini_utils import *
from mini_diffusion_utils import *
# from model import *
# from utils import *
# from diffusion_utils import *

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
        self.vis_step = config.diffusion.vis_step  # 100
        self.num_figs = config.diffusion.num_figs  # 10
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

        ema_helper = EMA(mu=self.config.model.ema_rate)
        ema_helper.register(model)

        if (
            config.diffusion.conditioning_signal == "NN"
            and config.diffusion.nonlinear_guidance.pre_train
        ):
            n_guidance_model_pretrain_epochs = (
                config.diffusion.nonlinear_guidance.n_pretrain_epochs
            )
            self.cond_pred_model.train()
            pretrain_start_time = time.time()
            for epoch in range(n_guidance_model_pretrain_epochs):
                self.nonlinear_guidance_model_train_loop_per_epoch(
                    train_loader, aux_optimizer, epoch
                )
            pretrain_end_time = time.time()
            logging.info(
                "Pre-training of non-linear guidance model took {:.4f} minutes.".format(
                    (pretrain_end_time - pretrain_start_time) / 60
                )
            )

            logging.info("\nAfter pre-training:")
            self.evaluate_guidance_model_on_both_train_and_test_set(
                dataset_object, train_loader, test_set_object, test_loader
            )
            # save auxiliary model
            aux_states = [
                self.cond_pred_model.state_dict(),
                aux_optimizer.state_dict(),
            ]
            torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

        # train diffusion model
        start_epoch, step = 0, 0

        if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
            logging.info("Prior distribution at timestep T has a mean of 0.")

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, xy_0 in enumerate(train_loader):
                n = xy_0.size(0)  # maybe batch size?
                data_time += time.time() - data_start
                model.train()
                step += 1

                # antithetic sampling -- low (inclusive) and high (exclusive)
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)

                # make sure the length of t
                t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                # noise estimation loss
                xy_0 = xy_0.to(self.device)
                x_batch = xy_0[:, : -config.model.y_dim]  # shape: (batch_size,1)
                y_batch = xy_0[:, -config.model.y_dim :]  # shape: (batch_size, 1)

                # Computing the mean of y_T by using pretrained guiding function
                y_0_hat_batch = self.compute_guiding_prediction(
                    x_batch, method=config.diffusion.conditioning_signal
                )
                y_T_mean = y_0_hat_batch

                # generate noise
                e = torch.randn_like(y_batch).to(y_batch.device)

                # noised data (forward process)
                y_t_batch = q_sample(
                    y_batch,
                    y_T_mean,
                    self.alphas_bar_sqrt,
                    self.one_minus_alphas_bar_sqrt,
                    t,  # timestep
                    noise=e,
                )

                # predicted noise
                # output = model(x_batch, y_t_batch, y_T_mean, t)
                output = model(x_batch, y_t_batch, y_0_hat_batch, t)

                loss = (
                    (e - output).square().mean()  # Mean Square Error
                )  # use the same noise sample e during training to compute loss

                # print logging every 1000 steps
                if step % self.config.training.logging_freq == 0 or step == 1:
                    logging.info(
                        (
                            f"epoch: {epoch}, step: {step}, Noise Estimation loss: {loss.item()}, "
                            + f"data time: {data_time / (i + 1)}"
                        )
                    )

                # optimize diffusion model that predicts eps_theta
                optimizer.zero_grad()
                loss.backward()

                with contextlib.suppress(Exception):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )

                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                # optimize non-linear guidance model
                # if update guidance during training
                if (
                    config.diffusion.conditioning_signal == "NN"
                    and config.diffusion.nonlinear_guidance.joint_train
                ):
                    self.cond_pred_model.train()
                    # One optimization step of the non-linear guidance model that predits y_0_hat.
                    aux_loss = self.nonlinear_guidance_model_train_step(
                        x_batch, y_batch, aux_optimizer
                    )
                    if step % self.config.training.logging_freq == 0 or step == 1:
                        logging.info(
                            f"meanwhile, non-linear guidance model joint-training loss: {aux_loss}"
                        )

                # save diffusion model to self.args.log_path
                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    if step > 1:  # skip saving the initial ckpt
                        torch.save(
                            states,
                            os.path.join(self.args.log_path, f"ckpt_{step}.pth"),
                        )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                    # save auxiliary model, which is pretrained guidance
                    if (
                        hasattr(config.diffusion.nonlinear_guidance, "joint_train")
                        and config.diffusion.nonlinear_guidance.joint_train
                    ):
                        assert config.diffusion.conditioning_signal == "NN"
                        aux_states = [
                            self.cond_pred_model.state_dict(),
                            aux_optimizer.state_dict(),
                        ]
                        if step > 1:  # skip saving the initial ckpt
                            torch.save(
                                aux_states,
                                os.path.join(
                                    self.args.log_path, f"aux_ckpt_{step}.pth"
                                ),
                            )

                        torch.save(
                            aux_states,
                            os.path.join(self.args.log_path, "aux_ckpt.pth"),
                        )

                if step % self.config.training.validation_freq == 0 or step == 1:
                    with torch.no_grad():
                        # plot q samples
                        if epoch == start_epoch:
                            fig, axs = plt.subplots(
                                1,
                                self.num_figs + 1,  # 10 + 1
                                figsize=((self.num_figs + 1) * 8.5, 8.5),
                                clear=True,
                            )
                            # q samples at timestep 1
                            y_1 = (
                                q_sample(
                                    y_batch,
                                    y_T_mean,
                                    self.alphas_bar_sqrt,
                                    self.one_minus_alphas_bar_sqrt,
                                    torch.tensor([0]).to(self.device),
                                )
                                .detach()
                                .cpu()
                            )
                            axs[0].scatter(
                                x_batch.detach().cpu(), y_1, s=10, c="tab:red"
                            )
                            axs[0].set_title(
                                "$q(\mathbf{y}_{" + str(1) + "})$", fontsize=23
                            )
                            y_q_seq = []

                            # plot every 100 steps
                            for j in range(1, self.num_figs):
                                cur_t = j * self.vis_step  # j * 100
                                cur_y = (  # cur_y: current y
                                    q_sample(
                                        y_batch,
                                        y_T_mean,
                                        self.alphas_bar_sqrt,
                                        self.one_minus_alphas_bar_sqrt,
                                        torch.tensor([cur_t - 1]).to(self.device),
                                    )
                                    .detach()
                                    .cpu()
                                )
                                y_q_seq.append(cur_y)
                                axs[j].scatter(
                                    x_batch.detach().cpu(),
                                    cur_y,
                                    s=10,
                                    c="tab:red",
                                )

                                axs[j].set_title(
                                    "$q(\mathbf{y}_{" + str(cur_t) + "})$",
                                    fontsize=23,
                                )

                            # q samples at timestep T
                            y_T = (
                                q_sample(
                                    y_batch,
                                    y_T_mean,
                                    self.alphas_bar_sqrt,
                                    self.one_minus_alphas_bar_sqrt,
                                    torch.tensor([self.num_timesteps - 1]).to(
                                        self.device
                                    ),
                                )
                                .detach()
                                .cpu()
                            )
                            axs[self.num_figs].scatter(
                                x_batch.detach().cpu(), y_T, s=10, c="tab:red"
                            )
                            axs[self.num_figs].set_title(
                                "$q(\mathbf{y}_{" + str(self.num_timesteps) + "})$",
                                fontsize=23,
                            )

                            ax_list = [axs[j] for j in range(self.num_figs + 1)]
                            ax_list[0].get_shared_x_axes().join(ax_list[0], *ax_list)
                            ax_list[0].get_shared_y_axes().join(ax_list[0], *ax_list)
                            if config.testing.squared_plot:
                                for j in range(self.num_figs + 1):
                                    axs[j].set(aspect="equal", adjustable="box")
                            fig.savefig(
                                os.path.join(
                                    args.im_path,
                                    f"q_samples_T{self.num_timesteps}_{step}.png",
                                )
                            )

                        # plot p samples
                        fig, axs = plt.subplots(
                            1,
                            self.num_figs + 1,
                            figsize=((self.num_figs + 1) * 8.5, 8.5),
                            clear=True,
                        )
                        y_p_seq = p_sample_loop(
                            model,
                            x_batch,
                            y_0_hat_batch,
                            y_T_mean,
                            self.num_timesteps,
                            self.alphas,
                            self.one_minus_alphas_bar_sqrt,
                        )

                        # p samples at timestep 1
                        cur_y = y_p_seq[self.num_timesteps - 1].detach().cpu()
                        axs[0].scatter(
                            x_batch.detach().cpu(), cur_y, s=10, c="tab:blue"
                        )
                        axs[0].set_title("$p({z}_1)$", fontsize=23)
                        # kl = kld(y_1, cur_y)
                        # kl_y0 = kld(y_batch.detach().cpu(), cur_y)
                        axs[0].set_title("$p(\mathbf{y}_{1})$", fontsize=23)
                        # axs[0].set_xlabel(
                        #     'KL($q(y_t)||p(y_t)$)={:.2f}\nKL($q(y_0)||p(y_t)$)={:.2f}'.format(
                        #         kl, kl_y0), fontsize=20)
                        for j in range(1, self.num_figs):
                            cur_t = j * self.vis_step
                            cur_y = y_p_seq[self.num_timesteps - cur_t].detach().cpu()
                            # kl = kld(y_q_seq[j-1].detach().cpu(), cur_y)
                            # kl_y0 = kld(y_batch.detach().cpu(), cur_y)
                            axs[j].scatter(
                                x_batch.detach().cpu(),
                                cur_y,
                                s=10,
                                c="tab:blue",
                            )
                            axs[j].set_title(
                                "$p(\mathbf{y}_{" + str(cur_t) + "})$",
                                fontsize=23,
                            )
                            # axs[j].set_xlabel(
                            #     'KL($q(y_t)||p(y_t)$)={:.2f}\nKL($q(y_0)||p(y_t)$)={:.2f}'.format(
                            #         kl, kl_y0), fontsize=20)
                        # p samples at timestep T
                        cur_y = y_p_seq[0].detach().cpu()
                        axs[self.num_figs].scatter(
                            x_batch.detach().cpu(), cur_y, s=10, c="tab:blue"
                        )
                        axs[self.num_figs].set_title(
                            "$p({z}_\mathbf{prior})$", fontsize=23
                        )
                        # kl = kld(y_T, cur_y)
                        # kl_y0 = kld(y_batch.detach().cpu(), cur_y)
                        # axs[self.num_figs].set_xlabel(
                        #     'KL($q(y_t)||p(z)$)={:.2f}\nKL($q(y_0)||p(z)$)={:.2f}'.format(
                        #         kl, kl_y0), fontsize=20)
                        if step > 1:
                            ax_list = [axs[j] for j in range(self.num_figs + 1)]
                            ax_list[0].get_shared_x_axes().join(ax_list[0], *ax_list)
                            ax_list[0].get_shared_y_axes().join(ax_list[0], *ax_list)
                            # define custom 'xlim' and 'ylim' values
                            # custom_xlim = axs[0].get_xlim()
                            # custom_ylim = axs[0].get_ylim()
                            # plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
                            if config.testing.squared_plot:
                                for j in range(self.num_figs + 1):
                                    axs[j].set(aspect="equal", adjustable="box")
                        fig.savefig(
                            os.path.join(
                                args.im_path,
                                f"p_samples_T{self.num_timesteps}_{step}.png",
                            )
                        )

                    plt.close("all")
                data_start = time.time()

        # save the model after training is finished
        states = [
            model.state_dict(),
            optimizer.state_dict(),
            epoch,
            step,
        ]

        if self.config.model.ema:
            states.append(ema_helper.state_dict())

        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
        # save auxiliary model after training is finished
        aux_states = [
            self.cond_pred_model.state_dict(),
            aux_optimizer.state_dict(),
        ]
        torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
        # report training set RMSE if applied joint training
        if config.diffusion.nonlinear_guidance.joint_train:
            y_rmse_aux_model = self.evaluate_guidance_model(
                dataset_object, train_loader
            )
            logging.info(
                "After joint-training, non-linear guidance model unnormalized y RMSE is {:.8f}.".format(
                    y_rmse_aux_model
                )
            )
