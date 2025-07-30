import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import scipy.linalg


class RobotArm:
    """Defines the kinematics of a 2-DOF robot arm."""
    def __init__(self, l1=1.0, l2=1.0):
        self.l1 = l1
        self.l2 = l2

    def forward_kinematics(self, theta):
        """Forward Kinematics: (theta1, theta2) -> (x, y)"""
        theta1, theta2 = theta[:, 0], theta[:, 1]
        x = self.l1 * torch.cos(theta1) + self.l2 * torch.cos(theta1 + theta2)
        y = self.l1 * torch.sin(theta1) + self.l2 * torch.sin(theta1 + theta2)
        return torch.stack([x, y], dim=1)

class AnosovSystem(nn.Module):
    def __init__(self):
        super().__init__()
        M = torch.tensor([[2.0, 1.0], [1.0, 1.0]])
        self.A = torch.tensor(scipy.linalg.logm(M.numpy()).real, dtype=torch.float32)
    def forward(self, t, y): 
        return torch.matmul(self.A, y.T).T

class StableSystem(nn.Module):
    """A simple, stable system where all points converges to a single sink at (pi, pi)"""
    def forward(self, t, y):
        return -0.5* (y-np.pi)



def generate_trajectory_data(robot, n_points=200, t_max=2.0):
    """Generates the ground truth trajectory data for the robot arm to draw a circle."""
    t = torch.linspace(0, t_max, n_points)
    # Target trajectory: a circle with center (1, 0) and radius 0.5
    center_x, center_y, radius = 1.0, 0.0, 0.5
    x_target = center_x + radius * torch.cos(2 * np.pi * t / t_max)
    y_target = center_y + radius * torch.sin(2 * np.pi * t / t_max)
    
    # Simplified Inverse Kinematics (IK)
    theta1_truth = torch.atan2(y_target, x_target)
    theta2_truth = torch.acos(torch.clamp((x_target**2 + y_target**2 - robot.l1**2 - robot.l2**2) / (2 * robot.l1 * robot.l2), -1, 1))
    
    # Normalize angles to the [0, 2pi) range
    theta_truth = torch.stack([theta1_truth, theta2_truth], dim=1) % (2 * np.pi)
    return t, theta_truth



def generate_chaotic_data(system, n_traj=10, n_points=50, t_max=1.0):
    t = torch.linspace(0, t_max, n_points)
    initial_points = torch.rand(n_traj, 2) * 2 * np.pi
    with torch.no_grad():
        trajectories = odeint(system, initial_points, t).permute(1, 0, 2)
    return t, trajectories

# In data_generation.py

def generate_biased_stable_data(system, n_trajectories=20, n_points=50, t_max=2.0, 
                                region_size=np.pi/2, noise_level=0.1):
    """
    Generates training data from the stable system, but only from a NARROW and NOISY region.
    """
    print(f"Generating biased data from a narrow region with noise (level={noise_level})...")
    t = torch.linspace(0, t_max, n_points)
    
    # Sample initial points from a much smaller region, e.g., [0, pi/2] x [0, pi/2]
    initial_points = torch.rand(n_trajectories, 2) * region_size
    
    with torch.no_grad():
        # Generate clean trajectories
        clean_trajectories = odeint(system, initial_points, t).permute(1, 0, 2)
        # Add noise
        noise = torch.randn_like(clean_trajectories) * noise_level
        noisy_trajectories = clean_trajectories + noise
        
    return t, noisy_trajectories