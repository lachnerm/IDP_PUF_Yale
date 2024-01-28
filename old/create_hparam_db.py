import optuna

optuna.create_study(
    study_name="optim",
    storage="sqlite:///optim.db",
    direction="maximize"
)
