import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn
from tqdm.auto import tqdm


class DenseStack(nn.Module):
    """
    Fully connected neural network.

    Args:
        config: Configparser section proxy with:
            num_in_features: Number of input features
            num_out_features: Number of output features
            num_hidden_features: List of nodes in each hidden layer
            use_batch_norm: If to use batch norm
            dropout_rate: If, and with which rate, to use dropout
    """

    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.fc_layers = []
        self.acts = []

        in_features = input_size
        # List containing number of hidden and output neurons
        list_of_out_features = [*hidden_size, output_size]
        for out_features in list_of_out_features:
            # Add fully connected layer
            self.fc_layers.append(nn.Linear(in_features, out_features))
            # Add activation function
            self.acts.append(nn.GELU())
            in_features = out_features
            self.num_out_features = out_features

        # Transform to pytorch list modules
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.acts = nn.ModuleList(self.acts)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fully connected neural network.

        Args:
            input_tensor: Tensor with input features

        Returns:
            Output prediction tensor
        """
        for i_layer in range(len(self.fc_layers)):
            # Fully connected layer
            input_tensor = self.fc_layers[i_layer](input_tensor)
            # Apply activation function, but not after last layer
            if i_layer < len(self.fc_layers) - 1:
                input_tensor = self.acts[i_layer](input_tensor)
        return input_tensor


class ESN(nn.Module):
    """Taken from https://github.com/danieleds/TorchRC/blob/master/torch_rc/nn/esn.py."""

    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        hidden_size: list,
        output_size: int,
        scale_rec: float = 1 / 1.1,
        scale_in: float = 1.0 / 40.0,
        leaking_rate: float = 0.5,
        rec_rescaling_method: str = "specrad",  # Either "norm" or "specrad"
    ):
        super(ESN, self).__init__()

        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.output_size = output_size

        self.leaking_rate = leaking_rate

        # Reservoir
        W_in = torch.rand((reservoir_size, input_size)) - 1 / 2.0
        W_hat = torch.randn((reservoir_size, reservoir_size))

        W_in = scale_in * W_in
        W_hat = self.rescale_contractivity(W_hat, scale_rec, rec_rescaling_method)

        # Assign as buffers
        self.register_buffer("W_in", W_in)
        self.register_buffer("W_hat", W_hat)
        # self.readout = nn.Linear(reservoir_size, output_size, bias=True)
        self.readout = DenseStack(reservoir_size, hidden_size, output_size)

    @staticmethod
    def rescale_contractivity(W, coeff, rescaling_method):
        if rescaling_method == "norm":
            return W * coeff / W.norm()
        elif rescaling_method == "specrad":
            return W * coeff / (torch.linalg.eig(W)[0].abs().max())
        else:
            raise Exception("Invalid rescaling method used (must be either 'norm' or 'specrad')")

    def forward_reservoir(self, input, hidden):
        """
        input: (batch, input_size)
        hidden: (batch, hidden_size)
        output: (batch, hidden_size)
        """
        h_tilde = torch.tanh(torch.mm(input, self.W_in.t()) + torch.mm(hidden, self.W_hat.t()))
        h = (1 - self.leaking_rate) * hidden + self.leaking_rate * h_tilde
        return h

    def forward(self, input, h_0=None, return_states=False):
        batch = input.shape[0]

        if h_0 is None:
            h_0 = input.new_zeros((batch, self.reservoir_size))

        next_layer_input = input  # (sequence_length, batch, input_size)
        layer_outputs = []  # list of (batch, hidden_size)
        step_h = h_0
        for i in range(next_layer_input.shape[1]):
            x_t = next_layer_input[:, i]
            h = self.forward_reservoir(x_t, step_h)  # (batch, hidden_size)
            step_h = h
            if return_states:
                layer_outputs.append(h)
            else:
                layer_outputs.append(self.readout(h))
        h_n = step_h
        layer_outputs = torch.stack(layer_outputs, axis=1)
        return layer_outputs, h_n


class ESNModel:
    def __init__(
        self, dataloader_train, dataloader_val, network, learning_rate=0.05, offset=1, ridge_factor=5e-9, device=None
    ):
        if torch.cuda.is_available() and device is None:
            self.device = "cuda"
        elif not torch.cuda.is_available() and device is None:
            self.device = "cpu"
        else:
            self.device = device

        print("Using:", self.device)

        self.net = network.to(self.device)

        self.trainable_parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        print("Trainable parameters: " + str(self.trainable_parameters))

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        self.offset = offset
        self.ridge_factor = ridge_factor

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        self.criterion = torch.nn.MSELoss().to(self.device)

        self.train_loss = []
        self.val_loss = []

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5, min_lr=0.000001
        )

    def train(self, ridge=False):
        """Train model."""
        if ridge:
            x, y = torch.tensor(self.dataloader_train.dataset.input_data, dtype=torch.float64), torch.tensor(
                self.dataloader_train.dataset.output_data, dtype=torch.float64
            )
            out, _ = self.net(x.to(self.device), return_states=True)
            out = out[:, self.offset :]
            y = y[:, self.offset :]
            out_np = out.reshape(-1, out.shape[-1]).detach().cpu().numpy()
            y_np = y.reshape(-1, self.net.input_size).detach().cpu().numpy()

            clf = Ridge(alpha=self.ridge_factor)
            clf.fit(out_np, y_np)
            self.net.readout.fc_layers[0].weight = torch.nn.Parameter(
                torch.tensor(clf.coef_, dtype=torch.float64).to(self.device)
            )
            self.net.readout.fc_layers[0].bias = torch.nn.Parameter(
                torch.zeros_like(self.net.readout.fc_layers[0].bias).to(self.device)
            )
            sum_loss = self.criterion(self.net.readout(out), y.to(self.device)).detach().cpu().numpy()
            cnt = 1
        else:
            self.net.train()
            cnt, sum_loss = 0, 0
            for (x, y) in self.dataloader_train:
                self.optimizer.zero_grad()
                out, _ = self.net(x.to(self.device))
                loss = self.criterion(out[:, self.offset :], y[:, self.offset :].to(self.device))
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.detach().cpu().numpy()
                cnt += 1
            self.optimizer.zero_grad()
            self.scheduler.step(sum_loss / cnt)
        self.train_loss.append(sum_loss / cnt)
        return sum_loss / cnt

    def validate(self):
        """Validate model."""
        self.net.eval()
        cnt, sum_loss = 0, 0
        with torch.no_grad():
            for (x, y) in self.dataloader_val:
                out, _ = self.net(x.to(self.device))
                loss = self.criterion(out[:, self.offset :], y[:, self.offset :].to(self.device))
                sum_loss += loss.detach().cpu().numpy()
                cnt += 1
        self.val_loss.append(sum_loss / cnt)
        return sum_loss / cnt

    def integrate(self, x_0, T, h0=None):
        """Integrate single trajectory."""
        self.net.eval()

        if h0 is None:
            h0 = torch.zeros(1, self.net.reservoir_size).to(self.device)

        x_list = []
        h_list = []
        x_0 = x_0.to(self.device)
        x_list.append(x_0[0].squeeze().detach().cpu().numpy())
        h_list.append(h0.squeeze().detach().cpu().numpy())
        h_t = h0
        # Warmup
        for x_t in x_0:
            x_out, h_t = self.net.forward(x_t.unsqueeze(0).unsqueeze(0), h_0=h_t)
            x_list.append(x_out.squeeze().detach().cpu().numpy())
            h_list.append(h_t.squeeze().detach().cpu().numpy())

        # Autoregressive integration
        for _ in tqdm(range(T), position=0, leave=True):
            if len(x_out.size()) > 2:
                x_out, h_t = self.net.forward(x_out, h_0=h_t)
            else:
                x_out, h_t = self.net.forward(x_out.unsqueeze(0), h_0=h_t)
            x_list.append(x_out.squeeze().detach().cpu().numpy())
            h_list.append(h_t.squeeze().detach().cpu().numpy())
        return np.array(x_list), np.array(h_list)

    def save_network(self, name):
        """Save network weights and training loss history."""
        filename = name + "_reservoir_size_" + str(self.net.reservoir_size) + ".net"
        torch.save(self.net.state_dict(), filename)
        np.save(name + "_training_loss.npy", np.array(self.train_loss))
        np.save(name + "_validation_loss.npy", np.array(self.val_loss))
        return name

    def load_network(self, name):
        """Load network weights and training loss history."""
        filename = name + "_reservoir_size_" + str(self.net.reservoir_size) + ".net"
        self.net.load_state_dict(torch.load(filename))
        self.train_loss = np.load(name + "_training_loss.npy").tolist()
        self.val_loss = np.load(name + "_validation_loss.npy").tolist()


def progress(train_loss, val_loss):
    """Define progress bar description."""
    return "Train/Loss: {:.6f}  Val/Loss: {:.6f}".format(train_loss, val_loss)
