import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint

from analysis import generate_basis_loops

def create_performance_figure(robot, time_gt, theta_gt, theta_pred, filename="figure1_performance.pdf"):
    """
    Figure 1: Visualizes model performance with an improved layout.
    Top panel shows the full trajectory.
    Bottom panel shows four snapshots at different times.
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Define the grid layout: 2 rows, 4 columns
    # Main trajectory plot will span the entire top row
    ax_main = plt.subplot2grid((2, 4), (0, 0), colspan=4)
    # Snapshot plots will occupy the bottom row
    ax1 = plt.subplot2grid((2, 4), (1, 0))
    ax2 = plt.subplot2grid((2, 4), (1, 1))
    ax3 = plt.subplot2grid((2, 4), (1, 2))
    ax4 = plt.subplot2grid((2, 4), (1, 3))
    
    # --- Top Panel: 2D End-Effector Trajectory ---
    gt_path = robot.forward_kinematics(theta_gt).detach().numpy()
    pred_path = robot.forward_kinematics(theta_pred).detach().numpy()
    ax_main.plot(gt_path[:, 0], gt_path[:, 1], 'b:', lw=2, label='Ground Truth Path')
    ax_main.plot(pred_path[:, 0], pred_path[:, 1], 'r-', lw=1.5, label='Predicted Path')
    ax_main.set_title('(a) End-Effector Trajectory Comparison')
    ax_main.set_xlabel('X coordinate')
    ax_main.set_ylabel('Y coordinate')
    ax_main.legend()
    ax_main.set_aspect('equal')
    ax_main.grid(True)

    # --- Bottom Panels: Robot Arm Snapshots ---
    def get_arm_coords(theta_vec):
        """Helper function to get arm linkage coordinates."""
        t1, t2 = theta_vec[0], theta_vec[1]
        x1 = robot.l1 * np.cos(t1); y1 = robot.l1 * np.sin(t1)
        x2 = x1 + robot.l2 * np.cos(t1 + t2); y2 = y1 + robot.l2 * np.sin(t1 + t2)
        return [0, x1, x2], [0, y1, y2]
    
    snapshot_axes = [ax1, ax2, ax3, ax4]
    # Select 4 evenly spaced keyframes
    num_frames = len(time_gt)
    keyframe_indices = [0, num_frames // 3, (2 * num_frames) // 3, num_frames - 1]
    subplot_labels = ['(b)', '(c)', '(d)', '(e)']

    for i, ax in enumerate(snapshot_axes):
        frame_idx = keyframe_indices[i]
        gt_x, gt_y = get_arm_coords(theta_gt[frame_idx].numpy())
        pred_x, pred_y = get_arm_coords(theta_pred[frame_idx].numpy())
        
        ax.plot(gt_x, gt_y, 'bo--', lw=2, markersize=6, alpha=0.8, label='Ground Truth')
        ax.plot(pred_x, pred_y, 'ro-', lw=2, markersize=4, alpha=0.8, label='Predicted')
        
        ax.set_title(f'{subplot_labels[i]} Snapshot at t={time_gt[frame_idx]:.2f}s')
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
        ax.set_aspect('equal'); ax.grid(True)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Performance figure saved to {filename}")
    plt.close(fig)

def create_homology_figure(model, filename="figure2_homology.pdf"):
    """Figure 2: Visualizes the action of the learned map on the homology basis."""
    # Generate and propagate basis loops
    basis_loops_2d = generate_basis_loops(n_points=100)
    with torch.no_grad():
        transformed_loops_2d = [odeint(model, loop, torch.tensor([0.0, 1.0]))[-1] for loop in basis_loops_2d]

    # Convert to 3D for plotting
    def to_3d(theta_2d, R=2, r=1):
        t1, t2 = theta_2d[:, 0], theta_2d[:, 1]
        x = (R + r * torch.cos(t2)) * torch.cos(t1); y = (R + r * torch.cos(t2)) * torch.sin(t1)
        z = r * torch.sin(t2)
        return torch.stack([x, y, z], dim=1).numpy()
    basis_loops_3d = [to_3d(loop) for loop in basis_loops_2d]
    transformed_loops_3d = [to_3d(loop) for loop in transformed_loops_2d]

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the torus surface
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, 2 * np.pi, 50))
    x_surf = (2 + 1 * np.cos(v)) * np.cos(u)
    y_surf = (2 + 1 * np.cos(v)) * np.sin(u)
    z_surf = 1 * np.sin(v)
    ax.plot_surface(x_surf, y_surf, z_surf, color='cyan', alpha=0.2)

    # Plot original and transformed loops
    ax.plot(basis_loops_3d[0][:, 0], basis_loops_3d[0][:, 1], basis_loops_3d[0][:, 2], 'r:', label='Original $\gamma_1$')
    ax.plot(basis_loops_3d[1][:, 0], basis_loops_3d[1][:, 1], basis_loops_3d[1][:, 2], 'b:', label='Original $\gamma_2$')
    ax.plot(transformed_loops_3d[0][:, 0], transformed_loops_3d[0][:, 1], transformed_loops_3d[0][:, 2], 'r-', lw=3, label='Transformed $g(\gamma_1)$', alpha=0.8)
    ax.plot(transformed_loops_3d[1][:, 0], transformed_loops_3d[1][:, 1], transformed_loops_3d[1][:, 2], 'b-', lw=3, label='Transformed $g(\gamma_2)$', alpha=0.8)
    
    ax.set_title('Action of the Learned Map on Homology Basis')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.legend()
    
    # Set equal aspect ratio for a natural look
    all_points = np.concatenate(basis_loops_3d + transformed_loops_3d)
    max_val = np.abs(all_points).max()
    ax.set_xlim(-max_val, max_val); ax.set_ylim(-max_val, max_val); ax.set_zlim(-max_val, max_val)
    
    plt.savefig(filename) # Save as PDF
    print(f"Homology figure saved to {filename}")
    plt.close(fig)
    
    

def plot_chaotic_trajectories(t_gt, trajectories_gt, trajectories_pred, n_to_plot=3, filename="figure4_chaotic_tracking.pdf"):
    """
    Visualizes how well the model tracks chaotic trajectories ON THE TORUS SQUARE.
    This is achieved by applying a modulo operation.
    """
    fig, axes = plt.subplots(1, n_to_plot, figsize=(6 * n_to_plot, 5.5))
    if n_to_plot == 1: axes = [axes]
    
    fig.suptitle('Chaotic Trajectories on the Torus [0, 2π] x [0, 2π]', fontsize=16)

    for i in range(n_to_plot):
        ax = axes[i]
        # Apply modulo to project the paths back onto the torus square
        gt_path = trajectories_gt[i].detach().numpy() % (2 * np.pi)
        pred_path = trajectories_pred[i].detach().numpy() % (2 * np.pi)
        
        ax.plot(gt_path[:, 0], gt_path[:, 1], 'b:', label='Ground Truth', alpha=0.7)
        ax.plot(pred_path[:, 0], pred_path[:, 1], 'r-', label='Predicted', alpha=0.7)
        
        # Mark start and end points
        ax.plot(gt_path[0, 0], gt_path[0, 1], 'go', markersize=10, label='Start')
        ax.plot(gt_path[-1, 0], gt_path[-1, 1], 'ks', markersize=8, label='End (GT)')
        
        ax.set_title(f'Trajectory {i+1}')
        ax.set_xlabel('Theta 1 (mod 2π)')
        ax.set_ylabel('Theta 2 (mod 2π)')
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 2 * np.pi)
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    print(f"Chaotic trajectory tracking figure (on torus) saved to {filename}")
    plt.close(fig)