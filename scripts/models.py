from torch.nn import Linear, Tanh
import torch.nn as nn
import torch

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
        if output.shape[1] == 3:
            vx, vy, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]
            return vx, vy, p
        elif output.shape[1] == 2:
            vx, vy = output[:, 0:1], output[:, 1:2]
            return vx, vy
        elif output.shape[1] == 1:
            p = output[:, 0:1]
            return p
        else:
            raise ValueError("Unexpected output dimension")

class complexPINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, hidden_dim=128, num_layers=6):
        super(complexPINN, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        output = self.net(x)
        if output.shape[1] == 3:
            vx, vy, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]
            return vx, vy, p
        elif output.shape[1] == 2:
            vx, vy = output[:, 0:1], output[:, 1:2]
            return vx, vy
        elif output.shape[1] == 1:
            p = output[:, 0:1]
            return p
        else:
            raise ValueError("Unexpected output dimension")




# Dummy model that outputs all ones
class DummyModel1(nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return torch.ones((N, 1), device=x.device)  # Return tensor of ones

class DummyModel2(nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return torch.ones((N, 1), device=x.device), torch.zeros((N, 1), device=x.device)