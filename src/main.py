import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchdiffeq import odeint

from models import ToroidalODE, ComplexToroidalODE
from data_generation import RobotArm, AnosovSystem, StableSystem
from data_generation import generate_trajectory_data, generate_chaotic_data, generate_biased_stable_data
from analysis import run_homological_analysis

try:
    from plotting import create_performance_figure, create_homology_figure, plot_chaotic_trajectories, create_hidden_instability_figure, create_hidden_instability_figure_3d
except ImportError:
    print("Warning: plotting.py not found. Plotting functions will be disabled.")
    def create_performance_figure(*args, **kwargs): pass
    def create_homology_figure(*args, **kwargs): pass
    def plot_chaotic_trajectories(*args, **kwargs): pass


def set_seed(seed):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def train_model(model, t_gt, y_gt, epochs=2001, lr=1e-3, is_batch=False):
    """Main training function"""
    print(f"--- Starting Training (Seed: {torch.initial_seed()}) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    initial_points = y_gt[:, 0, :] if is_batch else y_gt[0]

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = odeint(model, initial_points, t_gt)
        if is_batch: 
            y_pred = y_pred.permute(1, 0, 2)
        diff = y_pred - y_gt
        loss = ((torch.atan2(torch.sin(diff), torch.cos(diff)))**2).mean()
        loss.backward(); optimizer.step()
        if epoch % (epochs // 4) == 0: print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
    print("Training finished.")
    return model


##### Experiments #####
def run_experiment_1(seed, epochs=2001, lr=1e-3):
    """
    Runs Experiment 1: Foundational Case Study.
    - Trains a model on the robot arm task.
    - Generates figures for the paper.
    - Performs and prints the homological analysis.
    """
    print(f"===== Running Experiment 1 with Seed: {seed} =====")
    set_seed(seed)

    # 1. Setup task and data
    model = ToroidalODE()
    arm = RobotArm()
    time_gt, theta_gt = generate_trajectory_data(arm)
    
    # 2. Train the model
    trained_model = train_model(model=model,
                                t_gt=time_gt,
                                y_gt=theta_gt,
                                epochs=epochs,
                                lr=lr)
    with torch.no_grad():
        theta_pred = odeint(trained_model, theta_gt[0], time_gt)
        
    # 3. Generate figures for the paper
    # Suffix the filenames with the seed to avoid overwriting
    perf_filename = f"../results/figure1_performance_seed{seed}.pdf"
    hom_filename = f"../results/figure2_homology_seed{seed}.pdf"
    
    create_performance_figure(arm, time_gt, theta_gt, theta_pred, filename=perf_filename)
    create_homology_figure(trained_model, filename=hom_filename)

    # 4. Run final analysis and print diagnosis
    homology_matrix, eigenvalues = run_homological_analysis(trained_model)
    print_diagnosis(homology_matrix, eigenvalues, seed)

# Experiment 2
def run_experiment_2(seed, epochs=4001, lr=1e-4):
    """Runs Experiment 2: Training on Chaotic Data."""
    print(f"\n===== Running Experiment 2 with Seed: {seed} =====")
    set_seed(seed)
    anosov = AnosovSystem()
    time_gt, traj_gt = generate_chaotic_data(anosov)
    model = ComplexToroidalODE()
    trained_model = train_model(model, time_gt, traj_gt, epochs, lr, is_batch=True)
    
    with torch.no_grad():
        traj_pred = odeint(trained_model, traj_gt[0], time_gt)
    # Analysis
    homology_matrix, eigenvalues = run_homological_analysis(trained_model)
    print_diagnosis(homology_matrix, eigenvalues, seed)
    
    homology_fig_filename = f"../results/figure3_homology_chaotic_seed{seed}.pdf"
    create_homology_figure(trained_model, filename=homology_fig_filename)

    chaos_perf_filename = f"../results/figure4_chaotic_tracking_seed{seed}.pdf"
    plot_chaotic_trajectories(time_gt, traj_gt, traj_pred, filename=chaos_perf_filename)

# 

def run_experiment_3(seed):
    """Runs Experiment 3: Detecting Hidden Global Instability."""
    print(f"\n===== Running Experiment 3 with Seed: {seed} =====")
    set_seed(seed)
    
    # 1. Setup the stable system and generate biased data from it
    stable_system = StableSystem()
    time_gt, trajectories_gt = generate_biased_stable_data(stable_system)
    
    # 2. Train a complex model on the biased data
    model = ComplexToroidalODE()
    trained_model = train_model(model, time_gt, trajectories_gt, epochs=300, lr=1e-2, is_batch=True)
    
    # 3. Create the new 2x2 visualization
    figure_filename = f"../results/figure5_hidden_instability_seed{seed}.pdf"
    create_hidden_instability_figure(stable_system, trained_model, filename=figure_filename)
    create_hidden_instability_figure_3d(stable_system, trained_model)
    # 4. Run our global homological analysis for the final diagnosis
    homology_matrix, eigenvalues = run_homological_analysis(trained_model)
    print_diagnosis(homology_matrix, eigenvalues, seed)


def print_diagnosis(matrix, eigenvalues, seed):
    """Prints the final analysis results."""
    print(f"\n--- Homological Analysis Results (Seed: {seed}) ---")
    print(f"Homology Matrix:\n{matrix}")
    print(f"Eigenvalues: {eigenvalues}")
    is_stable = np.all(np.isclose(np.abs(eigenvalues), 1.0) | np.isclose(eigenvalues, 0))
    if is_stable:
        print("Diagnosis: The learned control system is STABLE.")
    else:
        print("Diagnosis: The learned control system has CHAOTIC properties.")
    print("=========================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments for Homological Analysis of Neural ODEs.")
    parser.add_argument('--experiment', type=str, default='1', help='Which experiment to run (e.g., "1").')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    
    args = parser.parse_args()

    if args.experiment == '1':
        if args.seed is None:
            seeds = [42, 123, 2024, 888, 9999]
            for seed in seeds:
                run_experiment_1(seed)
        else:
            run_experiment_1(seed=args.seed)
            
    elif args.experiment == '2':
        run_experiment_2(seed=42)
        
    elif args.experiment == '2a':
        # Experiment 2A: Insufficient Training
        run_experiment_1(seed=5555, epochs=10)

    elif args.experiment == '2b':
        # Experiment 2B: Training on Chaotic Data
        run_experiment_2(seed=42)
    
    elif args.experiment == '3':
        # 
        run_experiment_3(seed=42)
    else:
        print(f"Experiment {args.experiment} not recognized.")
