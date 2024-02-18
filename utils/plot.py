from torch.utils.data import DataLoader

from .model import ESN, ESNModel


def load_model(dataset_train, dataset_val, config):
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
    model.load_network(config["PATH"] + "model_")
    return model
