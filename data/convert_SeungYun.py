import h5py
import numpy as np
import scipy.io
from matplotlib import pyplot as plt

grain_size = 'S'
use_mean_subracted_data = False
use_1over4 = True

region_size = 1

folder = f'SeungYun_{region_size}region'
file = f'28x28_Rand20k_cfgs.mat'
file_path = f"./{folder}/{file}"

data = scipy.io.loadmat(file_path)

with h5py.File(f"./{folder}/data_orig.h5", 'r') as orig:
    with h5py.File(f"./{folder}/data.h5", 'w') as fd:

        responses = np.array(orig['savedFrames'])[:, :, :, 0]
        # responses = np.swapaxes(responses, 0, 2)
        # responses = responses.astype(np.int16)

        fd["responses"] = responses

        challenges_bit = orig['savedLabels']
        bin_to_float = lambda c: [int(bit) for bit in str(c[0])[2:-1]]
        challenges = np.array([bin_to_float(challenge) for challenge in challenges_bit])
        fd["challenges"] = challenges

        fd["min"] = np.min(responses)
        fd["max"] = np.max(responses)
