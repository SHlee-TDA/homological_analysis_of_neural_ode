import numpy as np
import torch
from torchdiffeq import odeint

def generate_basis_loops(n_points=100):
    """
    Generates the basis loops (gamma1, gamma2) for the 2-Torus.
    These represent the generators of the first homology group H_1(T^2).
    """
    theta = torch.linspace(0, 2 * np.pi, n_points)
    gamma1 = torch.stack([theta, torch.zeros_like(theta)], dim=1)
    gamma2 = torch.stack([torch.zeros_like(theta), theta], dim=1)
    return [gamma1, gamma2]

def run_homological_analysis(model, n_points=100):
    """Computes the homology transformation matrix and its eigenvalues for a trained model."""
    # 1. Generate basis loops using the dedicated function
    basis_loops = generate_basis_loops(n_points)
    
    # 2. Propagate loops through the learned map
    transformed_loops = []
    with torch.no_grad():
        for loop in basis_loops:
            transformed_loops.append(odeint(model, loop, torch.tensor([0.0, 1.0]))[-1])
            
    # 3. Construct homology matrix by calculating winding numbers
    matrix = np.zeros((2, 2), dtype=int)
    for i, loop in enumerate(transformed_loops):
        unwrapped = np.unwrap(loop.numpy(), axis=0)
        windings = (unwrapped[-1] - unwrapped[0]) / (2 * np.pi)
        matrix[:, i] = np.round(windings)
    
    # 4. Return matrix and eigenvalues for diagnosis
    eigenvalues = np.linalg.eigvals(matrix)
    return matrix, eigenvalues
