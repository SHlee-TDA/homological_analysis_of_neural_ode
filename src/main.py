import argparse
import torch
import numpy as np
import random
from torchdiffeq import odeint

from models import ToroidalODE
from data_generation import generate_trajectory_data, RobotArm
from analysis import run_homological_analysis
from plotting import create_performance_figure, create_homology_figure

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
    

def train_model(robot, t_gt, theta_gt, epochs=2001, lr=1e-3):
    """A slightly modified training function for the main script."""
    print(f"--- Starting Training (Seed: {torch.initial_seed()}) ---")
    model = ToroidalODE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        theta_pred_unwrapped = odeint(model, theta_gt[0], t_gt)
        diff = theta_pred_unwrapped - theta_gt
        loss = ((torch.atan2(torch.sin(diff), torch.cos(diff)))**2).mean()
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
            
    print("Training finished.")
    model.eval()
    with torch.no_grad():
        theta_pred = odeint(model, theta_gt[0], t_gt)
    
    return model, theta_pred

def run_experiment_1(seed):
    """
    Runs Experiment 1: Foundational Case Study.
    - Trains a model on the robot arm task.
    - Generates figures for the paper.
    - Performs and prints the homological analysis.
    """
    print(f"===== Running Experiment 1 with Seed: {seed} =====")
    set_seed(seed)

    # 1. Setup task and data
    arm = RobotArm()
    time_gt, theta_gt = generate_trajectory_data(arm)

    # 2. Train the model
    trained_model, theta_pred = train_model(arm, time_gt, theta_gt)

    # 3. Generate figures for the paper
    # Suffix the filenames with the seed to avoid overwriting
    perf_filename = f"../results/figure1_performance_seed{seed}.pdf"
    hom_filename = f"../results/figure2_homology_seed{seed}.pdf"
    
    create_performance_figure(arm, time_gt, theta_gt, theta_pred, filename=perf_filename)
    create_homology_figure(trained_model, filename=hom_filename)

    # 4. Run final analysis and print diagnosis
    homology_matrix, eigenvalues = run_homological_analysis(trained_model)
    
    print("\n--- Final Homological Analysis ---")
    print(f"Seed: {seed}")
    print(f"Homology Matrix:\n{homology_matrix}")
    print(f"Eigenvalues: {eigenvalues}")

    is_stable = np.all(np.isclose(np.abs(eigenvalues), 1.0) | np.isclose(eigenvalues, 0))
    if is_stable:
        print("Diagnosis: The learned control system is STABLE.")
    else:
        print("Diagnosis: The learned control system may have CHAOTIC properties.")
    print("=========================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments for Homological Analysis of Neural ODEs.")
    parser.add_argument('--experiment', type=str, default='1', help='Which experiment to run (e.g., "1").')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    args = parser.parse_args()

    if args.experiment == '1':
        run_experiment_1(seed=args.seed)
    # Add other experiments here later
    # elif args.experiment == '2a':
    #     run_experiment_2a(seed=args.seed)
    else:
        print(f"Experiment {args.experiment} not recognized.")
