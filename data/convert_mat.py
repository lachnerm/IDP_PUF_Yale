import argparse

import h5py
import numpy as np
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default="oprefl")
parser.add_argument('--file', default="CRPs_WithOPRefl.mat")
args = parser.parse_args()
folder = args.folder
file = args.file

file_path = f"./{folder}/{file}"
#c_data = scipy.io.loadmat(file_path)

with h5py.File(file_path, "r") as data:
    with h5py.File(f"./{folder}/data.h5", 'w') as fd:
        fd["challenges"] = np.array(
            [data.get(data["crp"]["challengev"][0][idx])[()].flatten() for idx
             in range(7751)])
        responses = np.array(
            [data.get(data["crp"]["response"][0][idx])[()] for idx in
             range(7751)])
        fd["responses"] = responses
        print(responses.shape)
        fd["min"] = np.min(responses)
        fd["max"] = np.max(responses)
