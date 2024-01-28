import optuna
from optuna.integration import \
    (
    PyTorchLightningPruningCallback
)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from experimental.Generator import PUFGenerator
from modules.DataModule import PUFDataModule
from utils.test_ids import test_ids_8k


def objective(trial):
    # Hyperparameter generation
    lr = trial.suggest_categorical("lr", [0.01, 0.005, 0.001, 0.0005, 0.0001])
    neuron_multiplier = trial.suggest_categorical('ns', [32, 64, 96])
    c_weight = trial.suggest_categorical('c_weight', [5, 7, 10])
    beta1 = trial.suggest_categorical('beta1', [0.5, 0.7, 0.9])
    beta2 = trial.suggest_categorical('beta2', [0.5, 0.7, 0.9, 0.999])

    hparams = {
        "bs": 16,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "ns": neuron_multiplier,
        "c_weight": c_weight
    }

    data_module = PUFDataModule(
        hparams["bs"], "8k1", training_ids, test_ids, do_crop=True,
        crop_size=512, bitwise=False, single_bit=False, is_1d=False
    )
    data_module.setup()

    logger = TensorBoardLogger('runs', name=f'runs/trial_{trial.number}')
    trainer = Trainer(
        gpus=1, max_epochs=300, logger=logger, num_sanity_val_steps=0,
        check_val_every_n_epoch=5,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="objective"),
            EarlyStopping(
                monitor="objective", min_delta=0.001, patience=5,
                check_on_train_epoch_end=False
            )
        ]
    )

    model = PUFGenerator(
        hparams, 100, "optim", data_module.denormalize, 512,
        False, "", do_log=True,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    return model.objective


'''
optuna.create_study(
    study_name="real-optim",
    storage="sqlite:///optim.db",
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=20, interval_steps=5
    )
)
'''

print("Starting optimization...")
study = optuna.load_study(
    study_name="real-optim",
    storage="sqlite:///optim.db"
)

test_ids = test_ids_8k
training_size = int(4000)
training_ids = list(
    set(list(range(8000))).symmetric_difference(set(test_ids)))[
               :training_size]

study.optimize(objective, n_trials=100)

pruned_trials = [t for t in study.trials if
                 t.state == optuna.structs.TrialState.PRUNED]
complete_trials = [t for t in study.trials if
                   t.state == optuna.structs.TrialState.COMPLETE]
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
