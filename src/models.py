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
    
    
class ComplexToroidalODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), 
            nn.Tanh(), 
            nn.Linear(128, 128), 
            nn.Tanh(),
            nn.Linear(128, 128), 
            nn.Tanh(), 
            nn.Linear(128, 128), 
            nn.Tanh(), 
            nn.Linear(128, 128), 
            nn.Tanh(), 
            nn.Linear(128, 2))
    def forward(self, t, y): return self.net(y)


class ODEFunc(nn.Moduel):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, t, h):
        # Autonomous model
        return self.net(h)
    

