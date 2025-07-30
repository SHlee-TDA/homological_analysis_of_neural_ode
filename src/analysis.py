import numpy as np
import torch
from torchdiffeq import odeint

def generate_basis_loops(n_dim = 2, n_points=100):
    """
    Generates the basis loops (gamma1, gamma2) for the 2-Torus (default).
    These represent the generators of the first homology group H_1(T^2).
    If n_dim > 2, generate the basis loops (gamma_n, ..., gamma_n)for the n-Torus.
    In this case, returns an array of shape (n_dim, n_points, n_dim).
    """
    if n_dim > 2:
        loops = []
        theta = torch.linspace(0, 2 * np.pi, n_points)
        for d in range(n_dim):
            loop = torch.zeros([n_points, n_dim])
            loop[:, d] = theta
            loops.append(loop)
    else:
        theta = torch.linspace(0, 2 * np.pi, n_points)
        gamma1 = torch.stack([theta, torch.zeros_like(theta)], dim=1)
        gamma2 = torch.stack([torch.zeros_like(theta), theta], dim=1)
        return [gamma1, gamma2]

def run_homological_analysis(model, n_points=200, n_dim=2):
    """Computes the homology transformation matrix and its eigenvalues for a trained model."""
    # 1. Generate basis loops using the dedicated function
    basis_loops = generate_basis_loops(n_dim=n_dim, n_points=n_points)
    
    # 2. Propagate loops through the learned map
    transformed_loops = []
    with torch.no_grad():
        for loop in basis_loops:
            # Compute loop transformed by 1-Time diffeomorphism g
            unit_time = torch.tensor([0.0, 1.0])
            transformed_loops.append(odeint(model, loop, unit_time)[-1])    # (n_points, n_dim)
            
    # 3. Construct homology matrix by calculating winding numbers
    matrix = np.zeros((n_dim, n_dim), dtype=int)
    for i, loop in enumerate(transformed_loops):
        # Remind! shape of loop (n_points, n_dim)
        unwrapped = np.unwrap(loop.numpy(), axis=0)
        windings = (unwrapped[-1] - unwrapped[0]) / (2 * np.pi)
        matrix[:, i] = np.floor(windings).astype(int)
    
    # 4. Return matrix and eigenvalues for diagnosis
    eigenvalues = np.linalg.eigvals(matrix)
    return matrix, eigenvalues
