import math
import torch
import numpy as np


def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule in ["cosine", "cosine_reverse"]:
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [
                min(
                    1
                    - (
                        math.cos(
                            ((i + 1) / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    )
                    / (
                        math.cos(
                            (i / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    ),
                    max_beta,
                )
                for i in range(num_timesteps)
            ]
        )
        if schedule == "cosine_reverse":
            betas = betas.flip(0)  # starts at max_beta then decreases fast
    return betas


# Forward functions
def q_sample(y, y_0_hat, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):
    """
    y_0_hat: prediction of pre-trained guidance model; can be extended to represent
        any prior mean setting at timestep T.
    """
    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    # q(y_t | y_0, x)
    return (
        sqrt_alpha_bar_t * y
        + (1 - sqrt_alpha_bar_t) * y_0_hat
        + sqrt_one_minus_alpha_bar_t * noise
    )


def p_sample_loop(
    model, x, y_0_hat, y_T_mean, n_steps, alphas, one_minus_alphas_bar_sqrt
):
    device = next(model.parameters()).device
    z = torch.randn_like(y_T_mean).to(device)
    cur_y = z + y_T_mean  # sample y_T
    y_p_seq = [cur_y]
    for t in reversed(range(1, n_steps)):  # t from T to 2
        y_t = cur_y
        cur_y = p_sample(
            model, x, y_t, y_0_hat, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt
        )  # y_{t-1}
        y_p_seq.append(cur_y)
    assert len(y_p_seq) == n_steps
    y_0 = p_sample_t_1to0(
        model, x, y_p_seq[-1], y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt
    )
    y_p_seq.append(y_0)
    return y_p_seq
