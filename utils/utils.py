import numpy as np
import torch
from PIL import Image
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssim
from torch import nn
from scipy.stats import pearsonr
from utils.gabor import gabor_hash


def calc_pc(real, gen):
    dims = [idx + 1 for idx in range(len(real.shape[1:]))]
    mx = torch.mean(real, dim=dims)
    my = torch.mean(gen, dim=dims)
    for _ in dims:
        mx = mx.unsqueeze(dim=-1)
        my = my.unsqueeze(dim=-1)
    vx = real - mx
    vy = gen - my
    num = torch.sum(vx * vy, dim=dims)
    den = torch.norm(vx, dim=dims) * torch.norm(vy, dim=dims)
    return num / den


def calc_nrmse(real, gen):
    # TODO: Pytorch implementation
    return torch.tensor([nrmse(x, y, normalization="euclidean") for x, y in
                         zip(real.cpu().numpy(), gen.cpu().numpy())])


def calc_ssim(real, gen):
    # TODO: Pytorch implementation
    return torch.tensor(
        [ssim(x, y) for x, y in zip(real.cpu().numpy(), gen.cpu().numpy())])


def calc_MAE(real, gen):
    return torch.tensor(
        [nn.L1Loss()(x, y) for x, y in zip(real, gen)])


def calc_fhd(real, gen, do_gabor=True):
    # TODO: Pytorch implementation
    real = real.squeeze().cpu().numpy()
    gen = gen.squeeze().cpu().numpy()
    fhds = []
    for real, gen in zip(real, gen):
        if do_gabor:
            real = gabor_hash(real)
            gen = gabor_hash(gen)
        real = real.flatten()
        gen = gen.flatten()
        fhd = np.count_nonzero(real != gen) / real.shape[0]
        fhds.append(fhd)

    return torch.tensor(fhds)

def calc_np_fhd(real, gen, do_gabor=True):
    fhds = []
    for real, gen in zip(real, gen):
        if do_gabor:
            real = gabor_hash(real)
            gen = gabor_hash(gen)
        real = real.flatten()
        gen = gen.flatten()
        fhd = np.count_nonzero(real != gen) / real.shape[0]
        fhds.append(fhd)

    return fhds

def calc_np_pc(real, gen, do_gabor=True):
    pcs = []
    for real, gen in zip(real, gen):
        pcs.append(pearsonr(real.flatten(), gen.flatten())[0])

    return pcs

def calc_bit_n_FHD(real, gen, n):
    n_idx = n - 1
    dims = [idx + 1 for idx in range(len(real.shape[1:]) - 1)]
    hd = (real[:, :, :, n_idx] != gen[:, :, :, n_idx]).sum(dim=dims)
    fhd = hd.float() / torch.tensor(real.shape[1] ** 2).float()
    return fhd


def store_preds(challenges, responses, store_path):
    with open(f"{store_path}/preds.npy", "wb") as f:
        challenges = challenges.squeeze().cpu().numpy()
        responses = responses.squeeze().cpu().numpy()
        np.savez(f, challenges=challenges, responses=responses)


def dec2bin_numpy(img, bits=14):
    img = img - img.min()
    bin_img = ((img.astype(int)[:, :, None] & (1 << np.arange(bits))[::-1]) > 0
               ).astype(int)
    return bin_img


def dec2bin_pytorch(inputs, bits=14):
    binary = []
    for img in inputs:
        img = (img - img.min()).int()
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(img.device, img.dtype)
        binary.append(img.unsqueeze(-1).bitwise_and(mask).ne(0).byte())
    return torch.stack(binary)


def bin2dec(inputs, bits=14):
    decimal = []
    for img in inputs:
        img = img.int()
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(img.device, img.dtype)
        decimal.append(torch.sum(mask * img, -1).float())
    return torch.stack(decimal)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def convert_stored_preds_to_imgs(store_path):
    with open(f"{store_path}/preds.npy", "rb") as f:
        data = np.load(f)
        responses = data["responses"]
        for idx in range(10):
            r = responses[idx]
            r = (r - np.min(r)) / (np.max(r) - np.min(r))
            r *= 255
            img = Image.fromarray(r).convert("L")
            img.save(f"{store_path}/{idx}", "jpeg")
