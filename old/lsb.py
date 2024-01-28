import json
import os

import h5py
import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange, tqdm


def dec2bin(r, bits):
    return (((r.astype(int)[:, None] & (1 << np.arange(bits))[
                                       ::-1])) > 0).astype(int)


def center_crop(img, crop_size=512):
    y_size, x_size = img.shape[-2:]
    x_start = x_size // 2 - (crop_size // 2)
    y_start = y_size // 2 - (crop_size // 2)
    if len(img.shape) == 2:
        return img[y_start:y_start + crop_size, x_start:x_start + crop_size]
    else:
        return img[:, y_start:y_start + crop_size, x_start:x_start + crop_size]


wanted_values = {
    "SSM": [0, 0.0234, 0.0285, 0.0882, 0.1724, 0.3202, 0.4497, 0.4970, 0.4999,
            0.5000, 0.5000, 0.5001, 0.5000, 0.5001],
    "SPM": [3.4809112548828124e-07, 0.06329127788543701, 0.08462510585784912,
            0.33041053771972656, 0.5380926084518433, 0.5200057601928711,
            0.49937445163726807, 0.49998451709747316, 0.5000236511230469,
            0.5000583505630494, 0.5000709962844848, 0.4999984264373779,
            0.49999418735504153, 0.4999891424179077],
    "DDM": [8.544921875000001e-08, 0.17135491943359374, 0.19055568847656246,
            0.4560382080078125, 0.49679927978515626, 0.49820761718749995,
            0.4978942504882813, 0.49797745361328133, 0.4979019775390625,
            0.4978639526367186, 0.49813009033203115, 0.4979535522460937,
            0.4980174194335937, 0.497964453125],
    "DPM": [1.9073486328125e-08, 0.06359298229217529, 0.08181410789489746,
            0.3013442468643188, 0.4939873456954956, 0.5145610570907593,
            0.5002708911895752, 0.4999509572982788, 0.5000261878967285,
            0.49999558448791503, 0.4999861192703247, 0.5000614643096923,
            0.5000205850601196, 0.4999602556228638]
}
data_orig = "data/8k1/data.h5"
data_ss = "data/cycle1/data.h5"
datasp = "results/8k1/tmp/preds/preds.npy"
datadd = "data/8k1/data.h5"
# datadp = open('g:\JoseJimenez\Pu\preds_D.mat')

bits = 14


def ss(crop_size):
    startIndex = 5
    endIndex = startIndex + 901
    range1 = trange(startIndex, endIndex)
    mean_fhd = []
    with h5py.File(data_ss, 'r') as data:
        for idx in range1:
            r1 = data["responses"][idx]
            if crop_size:
                r1 = center_crop(r1, crop_size)
            shifted_r1 = (r1 - np.min(r1)).flatten("F")
            r1_binary = dec2bin(shifted_r1, bits)

            fhd = []
            for idx2 in range(idx + 1000, 8001, 1000):
                r2 = data["responses"][idx2]
                if crop_size:
                    r2 = center_crop(r2, crop_size)
                shifted_r2 = (r2 - np.min(r2)).flatten("F")
                r2_binary = dec2bin(shifted_r2, bits)
                fhd_per_bit = []
                for bit in range(bits):
                    q1b = r1_binary[:, bit]
                    q2b = r2_binary[:, bit]
                    fhd_per_bit.append(
                        np.sum(q1b != q2b) / q1b.flatten().shape[0])
                fhd.append(fhd_per_bit)
            fhd = np.array(fhd)
            mean_fhd.append(fhd.mean(axis=0))

    mean_fhd_per_bit = np.mean(np.array(mean_fhd), axis=0)
    print(mean_fhd_per_bit)
    # plt.bar(x=list(range(1, 15)), height=wanted_values["SPM"], color="red")
    # plt.bar(x=list(range(1, 15)), height=mean_fhd_per_bit, color="blue")
    # plt.show()
    df = pd.DataFrame(mean_fhd)
    df.to_csv(f"pred_bits_ss.csv", index=False, header=False)
    print(df)


ss(512)
exit()


def sp(name, pred_data, crop_size=None):
    print("Starting SP for [", name, "/", crop_size, "]")
    mean_fhd = []
    with h5py.File(data_orig, 'r') as data:
        n = 800
        # n = 10
        for c, r_pred in tqdm(
                zip(pred_data["challenges"][:n], pred_data["responses"][:n]),
                total=n):
            c_idx = next(idx for idx, c2 in enumerate(data.get("challenges")) if
                         np.array_equal(c2, c))
            r = data.get("responses")[c_idx]
            r = center_crop(r)
            if crop_size is not None:
                r = center_crop(r, crop_size).astype(int)

            shifted_r = (r - np.min(r)).flatten("F")
            shifted_r_pred = (r_pred - np.min(r_pred)).flatten("F")

            r_binary = dec2bin(shifted_r, bits)
            r_pred_binary = dec2bin(shifted_r_pred, bits)

            fhd_per_bit = []
            for bit in range(bits):
                q1b = r_binary[:, bit]
                q2b = r_pred_binary[:, bit]
                fhd_per_bit.append(np.sum(q1b != q2b) / q1b.flatten().shape[0])
            mean_fhd.append(fhd_per_bit)

    mean_fhd_per_bit = np.mean(np.array(mean_fhd), axis=0)
    print(mean_fhd_per_bit)
    plt.bar(x=list(range(1, 15)), height=wanted_values["SPM"], color="red")
    plt.bar(x=list(range(1, 15)), height=mean_fhd_per_bit, color="blue")
    plt.show()
    df = pd.DataFrame(mean_fhd)
    df.to_csv(f"pred_bits_{name}.csv", index=False, header=False)
    print(df)


dir = "results_server/results/8k1"
print(next(os.walk(dir)))

_, folders, _ = next(os.walk("results_server/results/8k1"))
for folder in folders:
    preds = np.load(f"{dir}/{folder}/preds/preds_reg.npy")
    crop_size = int(folder.split("_csize")[1])
    sp(folder, preds, crop_size)

# datasp = "results/8k1/tmp_csize512/preds/preds_reg.npy"
# print("512 no crop - old data")
# sp()
with open("lsb_data.json", "r") as f:
    data = json.load(f)
data = {**data,
        "wanted_new": [0, 0.0234, 0.0285, 0.0882, 0.1724, 0.3202, 0.4497,
                       0.4970,
                       0.4999, 0.5000, 0.5000, 0.5001, 0.5000, 0.5001],
        "wanted_old": wanted_values["SSM"],
        "pred_old": wanted_values["SPM"]}
# exit()
'''datasp = "results/8k1/tmp_csize512/preds/preds_reg.npy"
print("512 - new data")
data_512_1 = sp(512)
data["512_1"] = data_512_1.tolist()

datasp = "results/8k1/tmp_csize512/preds/preds_reg_1.npy"
print("512 - new data 2")
data_512_2 = sp(512)
data["512_2"] = data_512_2.tolist()

datasp = "results/8k1/tmp_csize256/preds/preds_reg.npy"
print("256")
data_256 = sp(256)
data["256"] = data_256.tolist()

datasp = "results/8k1/tmp_csize128/preds/preds_reg.npy"
print("128")
data_128 = sp(128)
data["128"] = data_128.tolist()'''

with open("lsb_data.json", "w") as f:
    json.dump(data, f)

data_pd = {
    f"{dataset}_{idx, fhd}": (dataset, idx + 1, fhd)
    for dataset in data.keys()
    for idx, fhd in enumerate(data[dataset])
}

df = pandas.DataFrame.from_dict(
    data_pd,
    orient="index",
    columns=("Dataset", "Bit", "FHD")
)

df.to_csv("lsb_data.csv", index=False)


def dd():
    with h5py.File(datadd, 'r') as data:
        pairs = 250
        mean_fhd = []
        for idx in trange(pairs):
            idx1 = np.random.randint(0, 8000)
            idx2 = np.random.randint(0, 8000)
            if idx1 != idx2:
                r1 = data["responses"][idx1]
                shifted_r1 = (r1 - np.min(r1)).flatten("F")
                r1_binary = dec2bin(shifted_r1, bits)

                r2 = data["responses"][idx2]
                shifted_r2 = (r2 - np.min(r2)).flatten("F")
                r2_binary = dec2bin(shifted_r2, bits)

                fhd_per_bit = []
                for bit in range(bits):
                    q1b = r1_binary[:, bit]
                    q2b = r2_binary[:, bit]
                    fhd_per_bit.append(
                        np.sum(q1b != q2b) / q1b.flatten().shape[0])
                mean_fhd.append(fhd_per_bit)
            else:
                idx -= 1

    mean_fhd_per_bit = np.mean(np.array(mean_fhd), axis=0)
    print(mean_fhd_per_bit)
    plt.bar(x=list(range(1, 15)), height=mean_fhd_per_bit, color="blue")
    plt.bar(x=list(range(1, 15)), height=wanted_values["DDM"], color="red")
    plt.show()


def dp():
    mean_fhd = []
    pred_data = np.load(datadp)
    with h5py.File(datadd, 'r') as data:
        n = 800
        for c, r_pred in tqdm(
                zip(pred_data["challenges"][:n], pred_data["responses"][:n]),
                total=n):
            c_idx = next(idx for idx, c2 in enumerate(data.get("challenges")) if
                         np.array_equal(c2, c))
            r = data.get("responses")[c_idx]
            r = center_crop(r)

            shifted_r = (r - np.min(r)).flatten("F")
            shifted_r_pred = (r_pred - np.min(r_pred)).flatten("F")

            r_binary = dec2bin(shifted_r, bits)
            r_pred_binary = dec2bin(shifted_r_pred, bits)

            fhd_per_bit = []
            for bit in range(bits):
                q1b = r_binary[:, bit]
                q2b = r_pred_binary[:, bit]
                fhd_per_bit.append(np.sum(q1b != q2b) / q1b.flatten().shape[0])
            mean_fhd.append(fhd_per_bit)

    mean_fhd_per_bit = np.mean(np.array(mean_fhd), axis=0)
    print(mean_fhd_per_bit)
    plt.bar(x=list(range(1, 15)), height=wanted_values["DPM"], color="red")
    plt.bar(x=list(range(1, 15)), height=mean_fhd_per_bit, color="blue")
    plt.show()
