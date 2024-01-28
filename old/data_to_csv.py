import os
import numpy as np
import pandas as pd

dir = "results_server/results/8k1"
print(next(os.walk(dir)))
_, folders, _ = next(os.walk("results_server/results/8k1"))
pred_dict = {}
for folder in folders:
    preds = np.load(f"{dir}/{folder}/preds/preds_reg.npy")
    pred_dict[dir] = {"c": preds["challenges"], "r": preds["responses"]}
    break
df = pd.DataFrame(pred_dict)
print(df)
df.to_csv("preds.csv", index=False)