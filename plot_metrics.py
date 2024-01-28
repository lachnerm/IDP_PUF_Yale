import json
import os
import numpy as np
import pandas as pd


def build_results_multiple_sizes(result_folder, folders, sizes, metrics,
                                 r_test=False, tmp_folder=None):
    results = {
        dataset: {
            size: {
                metric: [] for metric in metrics
            } for size in sizes
        }
        for dataset in folders
    }

    for folder in folders:
        if not tmp_folder:
            tmp_folder = f"tmp{'_r_test' if r_test else ''}"
        folder_path = f"{result_folder}/{folder}/{tmp_folder}"
        print(folder_path)
        _, _, data_files = next(os.walk(folder_path))
        for data_file in data_files:
            if not 'json' in data_file:
                continue
            with open(f"{folder_path}/{data_file}", "r") as file:
                data = json.load(file)
                t_size = int(data_file.split("_")[0])
                #t_size = int(data_file.split("_")[1].split(".")[0])
                if len(data) > 0:
                    if t_size in sizes:
                        for metric in metrics:
                            if metric in data.keys():
                                results[folder][t_size][metric] += data[metric]
                            else:
                                print(
                                    f"{folder} doesn't have results for metric {metric}")
                else:
                    print(f"File {data_file} for folder {folder} is empty!")

    return results


def main():
    folders_exp = [
        "8k1",
        "8k2",
        "cycle1",
        "cycle2",
        "wl",
        "oprefl",
    ]
    folders_exp_new = [
        #"SeungYun_1region",
        #"SeungYun_4region",
        "SeungYun_16region",
    ]
    folders_sim = [
        "sim1",
        "sim2",
        "sim3",
        "sim4",
        "sim5",
        "Disordered",
        "Disordered_100",
        "Disordered_100_2"
    ]
    folders = folders_exp_new

    result_folder = "results_sv"

    metrics = ["PC", "FHD", "MAE"]

    sizes = [7200, 18000, 3686, 69750]

    results = build_results_multiple_sizes(
        result_folder, folders, sizes, metrics, False, tmp_folder="tmp"
    )

    results_pandas = {
        f"{idx}{data_set}{size}{metric}": (
            data_set, size, metric,
            results[data_set][size][metric][idx])
        for data_set in results.keys()
        for size in results[data_set].keys()
        for metric in results[data_set][size].keys()
        for idx in range(len(results[data_set][size][metric]))
    }

    df = pd.DataFrame.from_dict(
        results_pandas, orient="index",
        columns=("Dataset", "Training Size", "Metric", "Value")
    )

    #grouped_data = df[df['Training Size'] == size].groupby(['Dataset', 'Metric', 'Training Size'])
    grouped_data = df.groupby(['Dataset', 'Metric', 'Training Size'])
    pd.set_option("display.precision", 3)
    print("====================================================================================")
    print("MEAN:")
    print(grouped_data.mean())
    print("====================================================================================")
    print("STD:")
    print(grouped_data.std())
    """print("====================================================================================")
    print("MAD:")
    print(grouped_data.apply(lambda x: x.mad()))"""

if __name__ == "__main__":
    main()
