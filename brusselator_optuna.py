"""Run Brusselator example."""
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from brusselator.config import config
from brusselator.datasets import BrusselatorParallelDataset
from utils.model import ESN, ESNModel

torch.set_default_dtype(config["TRAINING"]["dtype"])

if not os.path.exists(config["PATH"]):
    os.makedirs(config["PATH"])

dataset_train = BrusselatorParallelDataset(
    config["DATA"]["n_train"], config["DATA"]["l_trajectories"], config["DATA"]["parameters"]
)
dataset_val = BrusselatorParallelDataset(
    config["DATA"]["n_val"], config["DATA"]["l_trajectories"], config["DATA"]["parameters"]
)
dataset_test = BrusselatorParallelDataset(
    config["DATA"]["n_test"], config["DATA"]["l_trajectories_test"], config["DATA"]["parameters"]
)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset_train.tt[:-1], dataset_train.input_data[0], label="u")
ax.plot(dataset_train.tt[:-1], dataset_train.v_data[0], label="v")
ax.set_xlabel("t")
plt.legend()
plt.savefig("brusselator/fig/data.pdf")
plt.close()


# Create PyTorch dataloaders for train and validation data
dataloader_train = DataLoader(
    dataset_train,
    batch_size=config["TRAINING"]["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=config["TRAINING"]["batch_size"],
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


def objective(trial):
    leaking_rate = trial.suggest_float("leaking_rate", 0.3, 0.7)
    reservoir_size = config["MODEL"]["reservoir_size"]
    ridge_factor = trial.suggest_int("ridge_factor", -3, 3, step=1)
    ridge_factor = 10**ridge_factor
    scale_rec = trial.suggest_float("scale_rec", 0.8, 1.0)
    scale_in = trial.suggest_float("scale_in", 0.0, 0.2)
    network = ESN(
        config["MODEL"]["input_size"],
        reservoir_size,
        config["MODEL"]["hidden_size"],
        config["MODEL"]["input_size"],
        scale_rec=scale_rec,
        scale_in=scale_in,
        leaking_rate=leaking_rate,
    )

    model = ESNModel(
        dataloader_train,
        dataloader_val,
        network,
        learning_rate=config["TRAINING"]["learning_rate"],
        offset=config["TRAINING"]["offset"],
        ridge_factor=ridge_factor,
        device=config["TRAINING"]["device"],
    )

    loss = model.train(ridge=config["TRAINING"]["ridge"])
    warmup = config["DATA"]["max_warmup"]
    predictions, _ = model.integrate(
        torch.tensor(dataset_test.input_data[0][:warmup], dtype=torch.get_default_dtype()).to(model.device),
        T=dataset_test.input_data[0].shape[0] - warmup - 1,
    )
    return np.mean((predictions[model.offset :] - dataset_test.input_data[0][model.offset :, 0]) ** 2)


study = optuna.create_study(storage="sqlite:///db.sqlite3", study_name="brusselator")  # Specify the storage URL here.

study.optimize(objective, n_trials=300)

print(study.best_params)
