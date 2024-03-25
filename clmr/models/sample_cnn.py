import torch.nn as nn
from torch import Tensor
from clmr.models.model import Model


class SampleCNN(Model):
    def __init__(self, strides: list[int], supervised: bool = False, out_dim: int = 0):
        super().__init__()

        self.strides = strides
        self.supervised = supervised
        self.sequential = [
            nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
        ]

        self.hidden = [
            [128, 128],
            [128, 128],
            [128, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 512],
        ]

        assert len(self.hidden) == len(
            self.strides
        ), "Number of hidden layers and strides are not equal"
        for stride, (h_in, h_out) in zip(self.strides, self.hidden):
            self.sequential.append(
                nn.Sequential(
                    nn.Conv1d(h_in, h_out, kernel_size=stride, stride=1, padding=1),
                    nn.BatchNorm1d(h_out),
                    nn.ReLU(),
                    nn.MaxPool1d(stride, stride=stride),
                )
            )

        # 1 x 512
        self.sequential.append(
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
        )

        self.sequential = nn.Sequential(*self.sequential)

        if self.supervised:
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(512, out_dim)

    def forward_features(self, x: Tensor) -> Tensor:
        out = self.sequential(x)
        if self.supervised:
            out = self.dropout(out)
        out = out.reshape(x.shape[0], out.size(1) * out.size(2))
        return out

    def forward(self, x: Tensor) -> Tensor:
        out = self.forward_features(x)
        logit = self.fc(out)
        return logit

