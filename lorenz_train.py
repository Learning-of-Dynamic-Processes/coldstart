"""Run Lorenz example."""
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lorenz.config import config
from lorenz.datasets import LorenzParallelDataset
from utils.model import ESN, ESNModel, progress

torch.set_default_dtype(config["TRAINING"]["dtype"])

if not os.path.exists(config["PATH"]):
    os.makedirs(config["PATH"])

dataset_train = LorenzParallelDataset(
    config["DATA"]["n_train"], config["DATA"]["l_trajectories"], config["DATA"]["parameters"], config["DATA"]["load_data"]
)
dataset_train.save_data()
dataset_val = LorenzParallelDataset(
    config["DATA"]["n_val"], config["DATA"]["l_trajectories"], config["DATA"]["parameters"], False
)
dataset_test = LorenzParallelDataset(
    config["DATA"]["n_test"], config["DATA"]["l_trajectories_test"], config["DATA"]["parameters"], False
)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset_train.tt[:-1], dataset_train.input_data[0], label="u")
ax.plot(dataset_train.tt[:-1], dataset_train.v_data[0, :, 0], label="v")
ax.plot(dataset_train.tt[:-1], dataset_train.v_data[0, :, 1], label="w")
ax.set_xlabel("t")
plt.legend()
plt.savefig("lorenz/fig/data.pdf")
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

network = ESN(
    config["MODEL"]["input_size"],
    config["MODEL"]["reservoir_size"],
    config["MODEL"]["hidden_size"],
    config["MODEL"]["input_size"],
    config["MODEL"]["scale_rec"],
    config["MODEL"]["scale_in"],
    config["MODEL"]["leaking_rate"],
)

model = ESNModel(
    dataloader_train,
    dataloader_val,
    network,
    learning_rate=config["TRAINING"]["learning_rate"],
    offset=config["TRAINING"]["offset"],
    ridge_factor=config["TRAINING"]["ridge_factor"],
    device=config["TRAINING"]["device"],
)

if config["TRAINING"]["ridge"]:
    model.train(ridge=config["TRAINING"]["ridge"])
else:
    # Train for the given number of epochs
    progress_bar = tqdm(
        range(0, config["TRAINING"]["epochs"]),
        leave=True,
        position=0,
        desc=progress(0, 0),
    )
    train_loss_list = []
    val_loss_list = []
    for _ in progress_bar:
        train_loss = model.train(ridge=config["TRAINING"]["ridge"])
        val_loss = model.validate()
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        progress_bar.set_description(progress(train_loss, val_loss))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_loss_list, label="train loss")
    ax.plot(val_loss_list, label="val loss")
    plt.legend()
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("")
    plt.show()

model.net = model.net.to("cpu")
model.save_network(config["PATH"] + "model_")
model.net = model.net.to(model.device)

warmup = config["DATA"]["max_warmup"]
predictions, _ = model.integrate(
    torch.tensor(dataset_test.input_data[0][:warmup], dtype=torch.get_default_dtype()).to(model.device),
    T=dataset_test.input_data[0].shape[0] - warmup - 1,
)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset_test.tt[:-1], dataset_test.input_data[0][:, 0], label="true")
if len(predictions.shape) > 1:
    ax.plot(dataset_test.tt[:-1], predictions[:, 0], label="prediction")
else:
    ax.plot(dataset_test.tt[:-1], predictions, label="prediction")
ax.axvline(x=dataset_test.tt[warmup], color="k")
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
plt.savefig("lorenz/fig/predictions.pdf")
plt.show()
