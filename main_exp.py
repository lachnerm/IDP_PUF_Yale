import argparse
import json
import os
import random
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from experimental.Generator import PUFGenerator
from modules.DataModule import PUFDataModule, PUFDataModuleVarSets
from numerical.Generator import PUFGenerator1D
from utils.test_ids import (
    test_ids_1k, test_ids_8k, test_ids_4k, test_ids_7k,
    test_ids_20k
)
from matplotlib import pyplot as plt
plt.switch_backend('agg')


def get_test_run_idx(root_folder, train_size):
    _, existing_runs, _ = next(os.walk(root_folder))
    dir_prefix = f"tmp{train_size}_r_test_"
    existing_runs = list(filter(lambda r: dir_prefix in r, existing_runs))
    r_idxs = list(map(lambda r: int(r.split(dir_prefix)[1]), existing_runs))
    return max(r_idxs) + 1 if len(r_idxs) > 0 else 0

def get_test_ids(set_size, random_test, test_size=None):
    if set_size == 1000:
        test_ids = test_ids_1k
    elif set_size == 4096:
        test_ids = test_ids_4k
    elif set_size == 7750:
        test_ids = test_ids_7k
    elif set_size == 20000:
        test_ids = test_ids_20k
    else:
        test_ids = test_ids_8k

    if random_test:
        ids = list(range(set_size))
        random.shuffle(ids)
        size = int(test_size * set_size) if test_size is not None else len(
            test_ids)
        test_ids = ids[:size]

    return test_ids


def run_regular_attack(bitwise, single_bit, pref_bit, c_bits, folder, hparams,
                       is_1d, logger_name, pd, root_folder, set_size, store,
                       random_test, train_size, test_size, load, **kwargs):
    r_idx = get_test_run_idx(root_folder, train_size) if random_test else 0
    tmp_folder = f"{root_folder}/tmp{train_size}{f'_r_test_{r_idx}' if random_test else ''}"
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    test_ids = get_test_ids(set_size, random_test, test_size=test_size)

    if store:
        store_path = f"{tmp_folder}/preds"
        Path(store_path).mkdir(parents=True, exist_ok=True)
    else:
        store_path = ""

    epochs = hparams["epochs"]
    training_size = int(set_size * train_size)
    training_ids = list(
        set(list(range(set_size))).symmetric_difference(set(test_ids)))[
                   :training_size]

    data_path = f"{tmp_folder}/results_{training_size}.json"
    data_module = PUFDataModule(
        hparams["bs"], folder, training_ids, test_ids, do_crop=not is_1d,
        bitwise=bitwise, single_bit=single_bit
    )
    data_module.setup()

    if is_1d:
        model = PUFGenerator1D(
            hparams, c_bits, logger_name, data_module.denormalize, store_path,
            do_log=not pd
        )
    else:
        r_bits = 14 if single_bit is None else 1
        model = PUFGenerator(
            hparams, c_bits, logger_name, data_module.denormalize, bitwise,
            store_path, do_log=not pd, r_bits=r_bits, pref_bit=pref_bit
        )
    if pd:
        trainer = Trainer(
            gpus=1, max_epochs=epochs, logger=False, num_sanity_val_steps=0,
            check_val_every_n_epoch=epochs + 1
        )
    else:
        logger_name = f"{folder}_single_run{'_r_test' if random_test else ''}" \
                      f"{f'_bit{single_bit}' if single_bit is not None else ''}" \
                      f"{f'_{logger_name}' if logger_name != 'unnamed' else ''}"
        logger = TensorBoardLogger('runs', name=logger_name)
        trainer = Trainer(gpus=1, max_epochs=epochs, logger=logger,
                          check_val_every_n_epoch=5)

    if load:
        model.load_state_dict(torch.load(f"{tmp_folder}/model.pt"))
    else:
        trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)
    results = model.results

    with open(data_path, 'w') as file:
        json.dump(results, file)

    return model, tmp_folder


def run_size_var_attack(bitwise, single_bit, c_bits, folder, hparams, is_1d,
                        logger_name, pd, root_folder, set_size, sv_sizes, store,
                        random_test, load, **kwargs):
    r_idx = get_test_run_idx(root_folder) if random_test else 0
    tmp_folder = f"{root_folder}/tmp{f'_r_test_{r_idx}' if random_test else ''}"
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    test_ids = get_test_ids(set_size, random_test)

    epochs = hparams["epochs"]
    if sv_sizes is None:
        training_sizes = [int(pctg * set_size) for pctg in
                          [0.1, 0.3, 0.5, 0.7, 0.9]]
    else:
        training_sizes = [int(size) for size in sv_sizes]
    for training_size in training_sizes:
        tmp_file = f'{root_folder}/tmp/{training_size}_results.json'
        if not os.path.isfile(tmp_file):
            with open(tmp_file, 'w') as file:
                json.dump([], file)

    for training_size in training_sizes:
        training_ids = list(
            set(list(range(set_size))).symmetric_difference(set(test_ids)))[
                       :training_size]

        data_path = f"{tmp_folder}/{training_size}_results.json"

        if store:
            store_path = f"{tmp_folder}/preds/{training_size}"
            Path(store_path).mkdir(parents=True, exist_ok=True)
        else:
            store_path = ""

        data_module = PUFDataModule(
            hparams["bs"], folder, training_ids, test_ids, do_crop=not is_1d,
            bitwise=bitwise, single_bit=single_bit
        )
        data_module.setup()

        if is_1d:
            model = PUFGenerator1D(
                hparams, c_bits, logger_name, data_module.denormalize,
                store_path, do_log=not pd
            )
        else:
            r_bits = 14 if single_bit is not None else 1
            model = PUFGenerator(
                hparams, c_bits, logger_name, data_module.denormalize, bitwise,
                store_path, do_log=not pd, r_bits=r_bits
            )
        if pd:
            trainer = Trainer(
                gpus=1, max_epochs=epochs, logger=False, num_sanity_val_steps=0,
                check_val_every_n_epoch=epochs + 1
            )
        else:
            new_logger_name = f"{folder}_{training_size}{'_r_test' if random_test else ''}" \
                              f"{f'_bit{single_bit}' if single_bit is not None else ''}" \
                              f"{f'_{logger_name}' if logger_name != 'unnamed' else ''}"
            logger = TensorBoardLogger(f'runs', name=new_logger_name)
            trainer = Trainer(gpus=1, max_epochs=epochs, logger=logger,
                              check_val_every_n_epoch=5)

        if load:
            model.load_state_dict(torch.load(f"{tmp_folder}/model.pt"))
        else:
            trainer.fit(model, datamodule=data_module)

        trainer.test(model, datamodule=data_module)
        results = model.results

        with open(data_path, 'w') as file:
            json.dump(results, file)

    return model, tmp_folder


def run_cycle_attack(bitwise, c_bits, folder, hparams, logger_name, pd,
                     root_folder, store, **kwargs):
    if store:
        print(
            "WARNING: Storing predictions is not implemented for cycle attacks!")
    tmp_folder = f"{root_folder}/tmp"
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    test_sizes = range(1, 8)
    for test_crp_idx in test_sizes:
        tmp_file = f'{tmp_folder}/c{test_crp_idx}_results.json'
        if not os.path.isfile(tmp_file):
            with open(tmp_file, 'w') as file:
                json.dump([], file)

    epochs = hparams["epochs"]
    training_ids = list(range(1000))
    start_test_ids = list(range(1000, 2000))

    data_module = PUFDataModule(
        hparams["bs"], folder, training_ids, start_test_ids, bitwise=bitwise
    )
    data_module.setup()

    model = PUFGenerator(
        hparams, c_bits, logger_name, data_module.denormalize, bitwise,
        store_path="", do_log=not pd
    )
    if pd:
        trainer = Trainer(
            gpus=1, max_epochs=epochs, logger=False, num_sanity_val_steps=0,
            check_val_every_n_epoch=epochs + 1
        )
    else:
        logger_name = f"{folder}{f'_{logger_name}' if logger_name != 'unnamed' else ''}"
        logger = TensorBoardLogger(f'runs', name=logger_name)
        trainer = Trainer(gpus=1, max_epochs=epochs, logger=logger,
                          check_val_every_n_epoch=5)

    trainer.fit(model, datamodule=data_module)

    for test_crp_idx in test_sizes:
        data_path = f"{tmp_folder}/c{test_crp_idx}_results.json"

        start = 1000 * test_crp_idx
        test_ids = list(range(start, start + 1000))
        data_module = PUFDataModule(
            hparams["bs"], folder, training_ids, test_ids, bitwise=bitwise
        )
        data_module.setup()
        trainer.test(model, datamodule=data_module)
        results = model.results

        with open(data_path, 'w') as file:
            json.dump(results, file)

    return model, tmp_folder


def run_var_sets_attack(bitwise, c_bits, folder, test_folder, hparams, is_1d,
                        max_train, max_test, logger_name, pd, root_folder,
                        store, **kwargs):
    tmp_folder = f"{root_folder}/tmp"
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)
    data_path = f"{tmp_folder}/varSet_results.json"

    epochs = hparams["epochs"]
    training_ids = list(range(8000))
    test_ids = list(range(8000))
    if max_train is not None:
        random.shuffle(training_ids)
        training_ids = training_ids[:max_train]
    if max_test is not None:
        random.shuffle(test_ids)
        test_ids = test_ids[:max_test]
    data_module = PUFDataModuleVarSets(
        hparams["bs"], folder, test_folder, training_ids, test_ids
    )
    data_module.setup()

    if store:
        store_path = f"{tmp_folder}/preds"
        Path(store_path).mkdir(parents=True, exist_ok=True)
    else:
        store_path = ""

    if is_1d:
        model = PUFGenerator1D(
            hparams, c_bits, logger_name, data_module.denormalize, store_path,
            do_log=not pd
        )
    else:
        model = PUFGenerator(
            hparams, c_bits, logger_name, data_module.denormalize, bitwise,
            store_path, do_log=not pd
        )
    if pd:
        trainer = Trainer(
            gpus=1, max_epochs=epochs, logger=False, num_sanity_val_steps=0,
            check_val_every_n_epoch=epochs + 1
        )
    else:
        new_logger_name = f"{folder}-{test_folder}_var_sets{f'_{logger_name}' if logger_name != 'unnamed' else ''}"
        logger = TensorBoardLogger(f'runs', name=new_logger_name)
        trainer = Trainer(gpus=1, max_epochs=epochs, logger=logger,
                          check_val_every_n_epoch=5)

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    results = model.results

    with open(data_path, 'w') as file:
        json.dump(results, file)

    return model, tmp_folder


def run_var_sets_sliced_attacks(bitwise, c_bits, folder, test_folder, hparams,
                                is_1d, logger_name, pd, root_folder, store,
                                **kwargs):
    all_training_ids = [
        list(range(1000)),
        list(range(4000, 5000)),
        list(range(7000, 8000))
    ]
    all_test_ids = all_training_ids
    tmp_folder = f"{root_folder}/tmp"
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    epochs = hparams["epochs"]

    for train_idx, training_ids in enumerate(all_training_ids):
        for test_idx, test_ids in enumerate(all_test_ids):
            data_path = f"{tmp_folder}/train{train_idx}_test{test_idx}_results.json"
            data_module = PUFDataModuleVarSets(
                hparams["bs"], folder, test_folder, training_ids, test_ids
            )
            data_module.setup()

            if store:
                store_path = f"{tmp_folder}/preds"
                Path(store_path).mkdir(parents=True, exist_ok=True)
            else:
                store_path = ""

            if is_1d:
                model = PUFGenerator1D(
                    hparams, c_bits, logger_name,
                    data_module.denormalize, store_path,
                    do_log=not pd
                )
            else:
                model = PUFGenerator(
                    hparams, c_bits, logger_name,
                    data_module.denormalize, bitwise, store_path,
                    do_log=not pd
                )
            if pd:
                trainer = Trainer(
                    gpus=1, max_epochs=epochs, logger=False,
                    num_sanity_val_steps=0, check_val_every_n_epoch=epochs + 1
                )
            else:
                new_logger_name = f"{folder}-{test_folder}_var_sets_sliced{f'_{logger_name}' if logger_name != 'unnamed' else ''}"
                logger = TensorBoardLogger(f'runs', name=new_logger_name)
                trainer = Trainer(gpus=1, max_epochs=epochs, logger=logger,
                                  check_val_every_n_epoch=5)

            trainer.fit(model, datamodule=data_module)
            trainer.test(model, datamodule=data_module)
            results = model.results

            with open(data_path, 'w') as file:
                json.dump(results, file)

    return model, tmp_folder


def run_iterative_attack(bitwise, c_bits, folder, hparams, is_1d, logger_name,
                         pd, root_folder, store, **kwargs):
    tmp_folder = f"{root_folder}/tmp"
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)

    epochs = hparams["epochs"]

    training_sizes = list(range(1000, 8000, 1000))

    for training_size in training_sizes:
        tmp_file = f'{root_folder}/tmp/{training_size}_results.json'
        if not os.path.isfile(tmp_file):
            with open(tmp_file, 'w') as file:
                json.dump([], file)

    for training_size in training_sizes:
        ids = list(range(8000))
        random.shuffle(ids)
        training_ids = ids[:training_size]
        test_ids = ids[training_size:]

        data_path = f"{tmp_folder}/{training_size}_results.json"

        if store:
            store_path = f"{tmp_folder}/preds/{training_size}"
            Path(store_path).mkdir(parents=True, exist_ok=True)
        else:
            store_path = ""

        data_module = PUFDataModule(
            hparams["bs"], folder, training_ids, test_ids, bitwise=bitwise
        )
        data_module.setup()

        if is_1d:
            model = PUFGenerator1D(
                hparams, c_bits, logger_name, data_module.denormalize,
                store_path, do_log=not pd
            )
        else:
            model = PUFGenerator(
                hparams, c_bits, logger_name, data_module.denormalize, bitwise,
                store_path, do_log=not pd
            )
        if pd:
            trainer = Trainer(
                gpus=1, max_epochs=epochs, logger=False, num_sanity_val_steps=0,
                check_val_every_n_epoch=epochs + 1
            )
        else:
            new_logger_name = f"{folder}_iter_{training_size}{f'_{logger_name}' if logger_name != 'unnamed' else ''}"
            logger = TensorBoardLogger(f'runs', name=new_logger_name)
            trainer = Trainer(gpus=1, max_epochs=epochs, logger=logger,
                              check_val_every_n_epoch=5)

        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)
        results = model.results

        with open(data_path, 'w') as file:
            json.dump(results, file)

    return model, tmp_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', '--data_file', required=True)
    parser.add_argument('--f2', '--data_file2')
    parser.add_argument('--name', default="unnamed")
    parser.add_argument('--load', action="store_true")
    parser.add_argument('--pd', action="store_true")
    parser.add_argument('--type', default="default")
    parser.add_argument('--store', action="store_true")
    parser.add_argument('--store-model', action="store_true")
    parser.add_argument('--is-1d', action="store_true")
    parser.add_argument('--r-test', action="store_true")
    parser.add_argument('--bitwise', action="store_true")
    parser.add_argument('--single-bit', type=int, default=None)
    parser.add_argument('--pref-bit', type=int, default=None)

    parser.add_argument('--cbits', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train-size', type=float, default=0.9)
    parser.add_argument('--test-size', type=float, default=None)
    parser.add_argument('--set-size', type=int, default=8000)
    parser.add_argument('--max-train', type=int, default=None)
    parser.add_argument('--max-test', type=int, default=None)
    parser.add_argument('--sv-sizes', nargs="+", default=None)

    args = parser.parse_args()
    folder = args.f
    run_type = args.type

    print("_______________________________________________________")
    print(
        f"Starting run on data file {folder} and type {run_type} \n"
        f"Mode {'PROD' if args.pd else 'DEV'}\n"
        f"Bitwise {'YES' if args.bitwise else 'NO'}"
        f"{f' on bit {args.single_bit}' if args.single_bit is not None else ' on all bits'}"
        f"{f' preferring bit {args.pref_bit}' if args.pref_bit is not None else ''}\n"
        f"Random test selection {'YES' if args.r_test else 'NO'}\n"
        f"Store Model {'YES' if args.store_model else 'NO'}\n"
        f"Store Responses {'YES' if args.store else 'NO'}\n"
        f"Custom train size {args.train_size if args.train_size else 'NONE'}\n"
        f"Custom test size {args.test_size if args.test_size else 'NONE'}\n"
        f"Custom set size {args.set_size if args.set_size else 'NONE'}\n"
        f"Max train size {args.max_train if args.max_train else 'NONE'}\n"
        f"Max test size {args.max_test if args.max_test else 'NONE'}\n"
        f"Custom sv sizes {args.sv_sizes if args.sv_sizes else 'NONE'}\n")
    print("_______________________________________________________")

    with open("hparams.json", "r") as hparam_f:
        all_hparams = json.load(hparam_f)
        if args.is_1d:
            hparams = all_hparams["1D"]
        else:
            hparams = all_hparams["reg"]
    hparams["epochs"] = args.epochs

    run_type_str = f'_{run_type}' if run_type != 'default' else ''
    test_folder_str = f"_{args.f2}" if "var_sets" in run_type else ""
    root_folder = f"results{run_type_str}/{folder}{'bit' if args.bitwise else ''}{test_folder_str}"
    Path(root_folder).mkdir(parents=True, exist_ok=True)

    run_kwargs = {
        "bitwise": args.bitwise,
        "c_bits": args.cbits,
        "folder": folder,
        "test_folder": args.f2,
        "hparams": hparams,
        "is_1d": args.is_1d,
        "max_train": args.max_train,
        "max_test": args.max_test,
        "logger_name": args.name,
        "pd": args.pd,
        "pref_bit": args.pref_bit,
        "random_test": args.r_test,
        "root_folder": root_folder,
        "set_size": args.set_size,
        "single_bit": args.single_bit,
        "sv_sizes": args.sv_sizes,
        "store": args.store,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "load": args.load
    }
    if run_type == "default":
        model, tmp_folder = run_regular_attack(**run_kwargs)
    elif run_type == "sv":
        model, tmp_folder = run_size_var_attack(**run_kwargs)
    elif run_type == "cycle":
        model, tmp_folder = run_cycle_attack(**run_kwargs)
    elif run_type == "var_sets":
        model, tmp_folder = run_var_sets_attack(**run_kwargs)
    elif run_type == "var_sets_sliced":
        model, tmp_folder = run_var_sets_sliced_attacks(**run_kwargs)
    elif run_type == "iter":
        model, tmp_folder = run_iterative_attack(**run_kwargs)
    else:
        print("Unknown run type", run_type)
        exit(0)

    if args.store_model:
        torch.save(model.state_dict(), f"{tmp_folder}/model.pt")


if __name__ == "__main__":
    main()
