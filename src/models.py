import torch
import torch.nn as nn
from torchdiffeq import odeint

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

    def forward(self, t, h):
        # y is a point on the torus; the network computes the velocity vector at y
        return self.net(h)
    
    
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
    def forward(self, t, h): 
        return self.net(h)


class ODEFunc(nn.Module):
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
    
class NeuralODE(nn.Module):
    def __init__(self, func: ODEFunc):
        super().__init__()
        self.func = func

    def forward(self, h, t_span):
        # integrate and return state at t_span[-1]
        traj = odeint(self.func, h, t_span, atol=1e-6, rtol=1e-6, method='rk4')
        return traj[-1]