import argparse

import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--folder', required=True)
parser.add_argument('--file', required=True)
args = parser.parse_args()
folder = args.folder
file = args.file

file_path = f"./{folder}/{file}"

with h5py.File(file_path, "r") as data:
    with h5py.File(f"./{folder}/data.h5", 'w') as fd:
        fd["challenges"] = np.array(
            [data.get(data["crp"]["challengev"][0][idx])[()].flatten() for idx
             in range(8000)])
        responses = np.array(
            [data.get(data["crp"]["response"][0][idx])[()] for idx in
             range(8000)])
        np.random.shuffle(responses)
        fd["responses"] = responses
        fd["min"] = np.min(responses)
        fd["max"] = np.max(responses)
