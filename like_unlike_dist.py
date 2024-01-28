import json

import h5py
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from modules.DataModule import center_crop
from utils.utils import calc_np_fhd, calc_np_pc, calc_fhd

dataset_number = 2
pred_type = '8k'
n_responses = 100

def plot_data():
    with h5py.File("data/Kyungduk_grainS_no_bg/data.h5", 'r') as data:
        rs = np.array(data.get("responses"))
        np.random.shuffle(rs)
        rs = rs[:n_responses]

        unlike_fhds = []
        unlike_pcs = []
        for idx, r1 in tqdm(enumerate(rs)):
            # prepare array of repeated r1s to compare with all following responses
            r2s = rs[idx + 1:]
            r1s = np.repeat(r1[None, :, :], r2s.shape[0], axis=0)

            fhds = calc_np_fhd(r1s, r2s, do_gabor=True)
            pcs = calc_np_pc(r1s, r2s)

            unlike_fhds.extend(fhds)
            unlike_pcs.extend(pcs)
    print(np.mean(unlike_fhds), len(unlike_fhds))
    print(np.mean(unlike_pcs), len(unlike_pcs))

    print("Starting to plot...")
    plt.hist(unlike_fhds, bins=100, label="Unlike FHD", density=True)
    plt.title("FHD Distribution (grainS)")
    plt.legend()
    plt.savefig("grainS_unlike_fhd_dist.png")
    plt.close()

    print("Starting to plot...")
    plt.hist(unlike_pcs, bins=100, label="Unlike PC", density=True)
    plt.title("PC Distribution (grainS)")
    plt.legend()
    plt.savefig("grainS_unlike_pc_dist.png")
    plt.close()

    with h5py.File("data/Kyungduk_grainL_no_bg/data.h5", 'r') as data:
        rs = np.array(data.get("responses"))
        np.random.shuffle(rs)
        rs = rs[:n_responses]
        unlike_fhds = []
        unlike_pcs = []
        for idx, r1 in tqdm(enumerate(rs)):
            # prepare array of repeated r1s to compare with all following responses
            r2s = rs[idx + 1:]
            r1s = np.repeat(r1[None, :, :], r2s.shape[0], axis=0)

            fhds = calc_np_fhd(r1s, r2s, do_gabor=True)
            pcs = calc_np_pc(r1s, r2s)

            unlike_fhds.extend(fhds)
            unlike_pcs.extend(pcs)

    print(np.mean(unlike_fhds), len(unlike_fhds))
    print(np.mean(unlike_pcs), len(unlike_pcs))
    print("Starting to plot...")
    plt.hist(unlike_fhds, bins=100, label="Unlike FHD", density=True)
    plt.title("FHD Distribution (grainL)")
    plt.legend()
    plt.savefig("grainL_unlike_fhd_dist.png")
    plt.close()

    print("Starting to plot...")
    plt.hist(unlike_pcs, bins=100, label="Unlike PC", density=True)
    plt.title("PC Distribution (grainL)")
    plt.legend()
    plt.savefig("grainL_unlike_pc_dist.png")
    plt.close()
    exit()

    # with open(f"distribution_data/pred_fhd_{pred_type}{dataset_number}.json", "r") as f:
    with open(f"distribution_data/pred_fhd_{pred_type}1.json", "r") as f:
        pred_fhds = json.load(f)

    # with open(f"distribution_data/pred_pc_{pred_type}{dataset_number}.json", "r") as f:
    with open(f"distribution_data/pred_pc_{pred_type}1.json", "r") as f:
        pred_pcs = json.load(f)

    with open(f"distribution_data/like_pc{dataset_number}.json", "r") as f:
        like_pcs = json.load(f)
    with open(f"distribution_data/unlike_pc{dataset_number}.json", "r") as f:
        unlike_pcs = json.load(f)
    with open(f"distribution_data/like_fhd{dataset_number}.json", "r") as f:
        unlike_fhds = json.load(f)
    with open(f"distribution_data/unlike_fhd{dataset_number}.json", "r") as f:
        unlike_fhds = json.load(f)

    plt.hist(unlike_fhds, bins=100, label="Like FHD", density=True)
    plt.hist(np.array(unlike_fhds)[np.array(unlike_fhds) > 0.1], bins=100,
             label="Unlike FHD", density=True)
    plt.hist(pred_fhds, bins=100, label="Prediction FHD", density=True)
    plt.title("FHD Distributions")
    plt.legend()
    plt.savefig(f"fhd_dist{dataset_number}.png")
    plt.clf()

    plt.hist(like_pcs, bins=100, label="Like PC", density=True)
    plt.hist(unlike_pcs, bins=100, label="Unlike PC", density=True)
    plt.hist(pred_pcs, bins=100, label="Prediction PC", density=True)
    plt.title("PC Distributions")
    plt.legend()
    plt.savefig(f"pc_dist{dataset_number}.png")
    exit()


plot_data()

with h5py.File(f"data/{pred_type}{dataset_number}/data.h5", 'r') as data:
    c = data.get("challenges")
    r = np.array(data.get("responses"))

    if pred_type == 'cycle':
        c_refs = c[:1000]
        r_refs = r[:1000]
    else:
        c_refs = c[()]
        r_refs = r[()]

if pred_type == 'cycle':
    preds = np.load(f'results/cycle{dataset_number}/preds/preds.npy')
elif pred_type == '8k':
    preds = np.load(f'results/8k{dataset_number}/tmp/preds/preds.npy')
c_preds = preds['challenges']
r_preds = preds['responses']

pred_fhds = []
pred_pcs = []
for c_pred, r_pred in tqdm(zip(c_preds, r_preds), total=800):
    idx = (c_refs == c_pred).all(axis=1).nonzero()
    r = r_refs[idx]
    r = center_crop(r)
    fhd = calc_np_fhd(r_pred[None, :, :], r, do_gabor=True)
    pc = calc_np_pc(r_pred[None, :, :], r, do_gabor=True)
    pred_fhds.extend(fhd)
    pred_pcs.extend(pc)

with open(f"distribution_data/pred_fhd_{pred_type}{dataset_number}.json",
          "w") as f:
    json.dump(pred_fhds, f)
with open(f"distribution_data/pred_pc_{pred_type}{dataset_number}.json",
          "w") as f:
    json.dump(pred_pcs, f)


def fhd_comp():
    like_pcs = []
    like_fhds = []
    for idx, r_batch1 in enumerate(r_refs):
        for idx2, r_batch2 in enumerate(r_refs[idx + 1:]):
            print(f'{idx} - {idx2}')
            like_fhd = calc_np_fhd(r_batch1, r_batch2, do_gabor=True)
            like_fhds.extend(like_fhd)
            like_pc = calc_np_pc(r_batch1, r_batch2, do_gabor=True)
            like_pcs.extend(like_pc)
    return like_fhds, like_pcs


def unlike_fhd_comp():
    unlike_fhds = []
    unlike_pcs = []
    for idx, r_batch1 in enumerate(r_refs):
        for idx2, r_batch2 in enumerate(r_refs[idx + 1:]):
            print(f'{idx} - {idx2}')
            np.random.shuffle(r_batch1)
            np.random.shuffle(r_batch2)
            unlike_pc = calc_np_pc(r_batch1, r_batch2, do_gabor=True)
            unlike_pcs.extend(unlike_pc)
            unlike_fhd = calc_np_fhd(r_batch1, r_batch2, do_gabor=True)
            unlike_fhds.extend(unlike_fhd)
    return unlike_fhds, unlike_pcs


like_fhds, like_pcs = fhd_comp()
unlike_fhds, unlike_pcs = unlike_fhd_comp()

with open(f"distribution_data/like_pc{dataset_number}.json", "w") as f:
    json.dump(like_pcs, f)

with open(f"distribution_data/unlike_pc{dataset_number}.json", "w") as f:
    json.dump(unlike_pcs, f)

with open(f"distribution_data/like_fhd{dataset_number}.json", "w") as f:
    json.dump(like_fhds, f)

with open(f"distribution_data/unlike_fhd{dataset_number}.json", "w") as f:
    json.dump(unlike_fhds, f)
