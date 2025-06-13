from torch.nn import Linear, Tanh
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
        Linear(2, 256),
        Tanh(),
        Linear(256, 3),
        )

    def forward(self, x):
        x = x.float()
        output = self.net(x)
        vx, vy, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]
        return vx, vy, p