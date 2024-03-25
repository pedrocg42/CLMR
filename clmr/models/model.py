import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def initialize(self, m):
        if isinstance(m, (nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
