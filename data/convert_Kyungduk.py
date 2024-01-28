import h5py
import numpy as np
import scipy.io
from matplotlib import pyplot as plt

grain_size = 'S'
use_mean_subracted_data = False
use_1over4 = True
use_1over16 = False

folder = f'Kyungduk_grain{grain_size}{"_subtract" if use_mean_subracted_data else ""}{"_1over4" if use_1over4 else ""}{"_1over16" if use_1over16 else ""}'
folder += '_open'
file = f'CRP_grain{grain_size}{"_1over4" if use_1over4 else ""}{"_1over16" if use_1over16 else ""}_open.mat'

file_path = f"./{folder}/{file}"

data = scipy.io.loadmat(file_path)

with h5py.File(f"./{folder}/data.h5", 'w') as fd:
    challenges = data['C'].reshape(-1, data['C'].shape[-1])
    fd["challenges"] = np.swapaxes(challenges, 0, 1)

    # 1over4 data has both normal and subtracted data
    #if use_1over4:
    responses = data['R_subtract']
    #else:
    #    responses = data['R']
    # raw data still has all 3 channels
    '''if use_mean_subracted_data:
        # all 3 channels are the same
        responses = responses[0]'''
    responses = np.swapaxes(responses, 0, 2)
    # responses = responses.astype(np.int16)
    plt.imshow(responses[0])
    plt.show()

    # subtract mean as estimate for background light
    '''if subtract_mean:
        mean_r = np.mean(responses, axis=0)
        for idx in trange(responses.shape[0]):
            r = responses[idx] - mean_r
            responses[idx] = np.around(r)'''

    fd["responses"] = responses
    fd["min"] = np.min(responses)
    fd["max"] = np.max(responses)
