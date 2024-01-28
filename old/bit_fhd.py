import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


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


real_data = "data/8k1/data.h5"
preds = np.load(f"results_box/preds_reg.npy")
crop_size = 512
bits = 14

mean_fhd = []
with h5py.File(real_data, 'r') as data:
    n = 800
    for c, r_pred in tqdm(
            zip(preds["challenges"][:n],
                preds["responses"][:n]),
            total=n):
        c_idx = next(
            idx for idx, c2 in enumerate(data.get("challenges")) if
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
            fhd_per_bit.append(
                np.sum(q1b != q2b) / q1b.flatten().shape[0])
        mean_fhd.append(fhd_per_bit)

df = pd.DataFrame(mean_fhd)
df.to_csv(f"pred_bit_fhd.csv", index=False, header=False)

plt.boxplot(x=np.vstack(mean_fhd))
plt.show()

mean_fhd_per_bit = np.mean(np.array(mean_fhd), axis=0)
plt.bar(x=list(range(1, 15)), height=mean_fhd_per_bit, color="blue")
plt.show()
