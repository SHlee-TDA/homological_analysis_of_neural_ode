import torch
import torch.nn as nn

class ToroidalODE(nn.Module):
    """A learnable Neural ODE model."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2) # The final output is a 2D vector representing angular velocities
        )

    def forward(self, t, y):
        # y is a point on the torus; the network computes the velocity vector at y
        return self.net(y)