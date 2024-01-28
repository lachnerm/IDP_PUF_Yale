import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from utils import utils


def get_gen_output_figure_1d(challenge, real_response, gen_response,
                             challenge_shape, response_shape):
    cols = 5
    rows = min(6, challenge.shape[0])
    fig, axs = plt.subplots(rows, cols, figsize=(16, 20))

    for row in range(rows):
        real = real_response[row].unsqueeze(dim=0)
        gen = gen_response[row].unsqueeze(dim=0)
        curr_challenge = challenge[row]

        '''pc = utils.calc_pc(real, gen)[0]
        mse_e = utils.calc_nrmse(real, gen)[0]
        title = f"PC {pc:.2f}, MSE_E {mse_e:.2f}"'''

        show_img_in_ax(axs[row, 0], curr_challenge.reshape(challenge_shape),
                       challenge_shape)
        show_img_in_ax(axs[row, 1], real, response_shape, snakify=True)
        show_img_in_ax(axs[row, 2], gen, response_shape, snakify=True,
        #               title=title)
                       title='')
        show_diff_map_in_ax_1d(axs[row, 3], real, gen, response_shape)
        show_diff_map_in_ax_1d(axs[row, 4], real, gen, response_shape,
                               scaled=True)
    return fig


def get_gen_output_figure_2d(challenge, real_response, gen_response,
                             challenge_shape, response_shape):
    cols = 5
    rows = min(6, challenge.shape[0])
    fig, axs = plt.subplots(rows, cols, figsize=(16, 20))

    for row in range(rows):
        real = real_response[row].squeeze().unsqueeze(dim=0)
        gen = gen_response[row].squeeze().unsqueeze(dim=0)
        curr_challenge = challenge[row]

        '''pc = utils.calc_pc(real, gen)[0]
        ssim = utils.calc_ssim(real, gen)[0]
        mse_e = utils.calc_nrmse(real, gen)[0]
        title = f"PC {pc:.2f}, SSIM {ssim:.2f}, MSE_E {mse_e:.2f}"'''

        if challenge_shape:
            show_img_in_ax(axs[row, 0], curr_challenge, challenge_shape)
            # Case with 3 additional challenge bits containing other meta-data
        else:
            curr_challenge = curr_challenge[:-3]
            bits = int(math.sqrt(curr_challenge.shape[0]))
            show_img_in_ax(axs[row, 0], curr_challenge, (bits, bits))

        show_img_in_ax(axs[row, 1], real, response_shape)
        #show_img_in_ax(axs[row, 2], gen, response_shape, title=title)
        show_img_in_ax(axs[row, 2], gen, response_shape, title='')
        show_diff_map_in_ax_2d(axs[row, 3], real, gen)
        show_diff_map_in_ax_2d(axs[row, 4], real, gen, scaled=True)
    return fig


def show_img_in_ax(ax, img, img_size, snakify=False, title=""):
    ax.axis("off")
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')

    if title:
        ax.set_title(title)

    img = img.cpu().numpy().squeeze()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    if snakify:
        img_copy = img.copy()
        img_copy.resize(img_size, refcheck=False)
        img_copy = np.reshape(img_copy, img_size)
        img_copy[1::2] = img_copy[1::2, ::-1]
        img = img_copy
    else:
        img = np.reshape(img, img_size)

    ax.imshow(img, cmap="gray", vmin=0, vmax=1)


def show_diff_map_in_ax_1d(ax, img1, img2, img_size, scaled=False):
    ax.axis("off")

    img1 = torch.squeeze(img1).cpu().numpy()
    img2 = torch.squeeze(img2).cpu().numpy()

    diff_map = np.absolute((img1 - img2))
    diff_map.resize(img_size, refcheck=False)
    diff_map = np.reshape(diff_map, img_size)
    diff_map[1::2] = diff_map[1::2, ::-1]

    if not scaled:
        min = np.min((img1, img2))
        max = np.max((img1, img2))
        ax.set_title(f"Abs Diff.")
        ax.imshow(diff_map, cmap="gray", vmin=min, vmax=max)
    else:
        ax.set_title(f"Abs Diff. (scaled)")
        ax.imshow(diff_map, cmap="gray")


def show_diff_map_in_ax_2d(ax, real_response, gen_response, scaled=False):
    ax.axis("off")

    real_response = torch.squeeze(real_response).cpu().numpy()
    real_response = (real_response - np.min(real_response)) / (
            np.max(real_response) - np.min(real_response))

    gen_response = torch.squeeze(gen_response).cpu().numpy()
    gen_response = (gen_response - np.min(gen_response)) / (
            np.max(gen_response) - np.min(gen_response))

    difference_map = np.absolute((real_response - gen_response))

    if not scaled:
        min = np.min((real_response, gen_response))
        max = np.max((real_response, gen_response))
        ax.set_title(f"Abs Diff.")
        ax.imshow(difference_map, cmap="gray", vmin=min, vmax=max)
    else:
        ax.set_title(f"Abs Diff. (scaled)")
        ax.imshow(difference_map, cmap="gray")


def plot_grad_flow(named_parameters, axis):
    '''
    Plots the gradients flowing through different layers in the net during training. Assumes that a figure was
    initiated beforehand.
    '''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    axis.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    axis.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    axis.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    axis.set_xticks(range(0, len(ave_grads), 1))
    axis.set_xticklabels(layers, rotation="vertical")
    axis.set_xlim(left=0, right=len(ave_grads))
    axis.set_ylim(bottom=-0.001, top=0.2)
    axis.set_xlabel("Layers")
    axis.set_ylabel("Average gradient")
    axis.grid(True)
    axis.legend([Line2D([0], [0], color="c", lw=4),
                 Line2D([0], [0], color="b", lw=4),
                 Line2D([0], [0], color="k", lw=4)],
                ['max-gradient', 'mean-gradient', 'zero-gradient'])


def create_gradient_figure(name):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_title(f"{name} Gradient flow")
    return fig, ax
