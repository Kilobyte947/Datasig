"""This file defines trainable regressors for the toy Lipschitz experiment."""

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}

class TinyMLP(nn.Module):
    """A simple feedforward MLP with a configurable number of hidden layers and units, and a choice of activation function. 
    The output layer is linear (no activation). The final output is squeezed to shape (N,) for N input points."""
    def __init__(self, input_dim, hidden_sizes=(64, 64), activation="tanh"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation}")
        act_cls = _ACTIVATIONS[activation]

        dims = [input_dim] + list(hidden_sizes)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act_cls())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SingleTanhUnit(nn.Module):
    """"A model that's architecturally forced to have exact shape as the Tier A's ground truth : f(x) = A * tanh(w^T x + b), A/w/b as learnable params. 
    Used only for tier A sanity check."""""

    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_dim) * 0.5)
        self.b = nn.Parameter(torch.zeros(1))
        self.A = nn.Parameter(torch.ones(1))

    def forward(self, x):
        z = (x * self.w).sum(dim=-1) + self.b
        return self.A * torch.tanh(z)


def train_regressor(model, x_train, y_train, epochs, lr, weight_decay=0.0, batch_size=None, seed=None):
    """ Plain gradient descent training (Adam optimizer, mean-squared-error loss).
    Note: `seed` argument only controls the randomness during training (e.g. shuffling for mini-batches) and not how model's weights are randomly initialized."""
    if seed is not None:
        torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    N = x_train.shape[0]
    loss_history = []

    for _ in range(epochs):
        if batch_size is None or batch_size >= N:
            optimizer.zero_grad()
            pred = model(x_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        else:
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                pred = model(x_train[idx])
                loss = loss_fn(pred, y_train[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            loss_history.append(epoch_loss / n_batches)

    return model, loss_history