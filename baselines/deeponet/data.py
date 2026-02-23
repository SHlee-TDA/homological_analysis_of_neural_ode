import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class IdealMassSpringDataset(Dataset):
    def __init__(self, 
                 mass=1.0, k=1.0, 
                 n_traj= 50, # = number of initial states
                 n_obs=30,   # = number of y
                 noise_std=0.0, 
                 t_span=(0, 3)
                 ):
        self.mass = mass
        self.k = k
        self.n_traj = n_traj
        self.n_obs = n_obs
        self.noise_std = noise_std
        self.t_span = t_span
        
        self.branch_inputs = []
        self.trunk_inputs = []
        self.targets = []
        
        self._create_dataset()
        
    def sample_init(self):
        E = np.random.uniform(0.2, 1.0)
        theta = np.random.uniform(0, 2*np.pi)
        q0 = np.sqrt(2*E/self.k) * np.cos(theta)
        p0 = np.sqrt(2*self.mass*E) * np.sin(theta)
        return np.array([q0, p0])
        
    def hamiltonian(self, q, p):
        return 0.5*self.k*q**2 + 0.5*(p**2)/self.mass
    
    def symplectic_gradient(self, q, p):
        dH_dp = p/self.mass
        dH_dq = self.k*q
        return np.array([dH_dp, -dH_dq])
    
    def generate_trajectory(self):
        z0 = self.sample_init()
        
        # Sample time points randomly within t_span
        t_eval = np.sort(np.random.uniform(self.t_span[0], self.t_span[1], self.n_obs))
        
        dydt = lambda t, z: self.symplectic_gradient(z[0], z[1])
        
        sol = solve_ivp(
            fun=dydt,
            t_span=self.t_span,
            y0=z0,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-10
        )

        zt = sol.y.T  # (time_steps, 2)
        
        # Data formatting
        
        # Branch input : copy initial state for all time steps (n_obs, 2)
        u_sample = np.tile(z0, (self.n_obs, 1))
        
        # Trunk input : time points (n_obs, 1)
        y_sample = t_eval.reshape(-1, 1)
        
        # Target : state zt (n_obs, 2)
        target_sample =  zt
        
        if self.noise_std > 0:
                target_sample += np.random.normal(0, self.noise_std, target_sample.shape)

        return u_sample, y_sample, target_sample
    
    def _create_dataset(self):
        for _ in range(self.n_traj):
            u_sample, y_sample, target_sample = self.generate_trajectory()
            self.branch_inputs.append(u_sample)
            self.trunk_inputs.append(y_sample)
            self.targets.append(target_sample)
        
        self.branch_inputs = torch.tensor(np.concatenate(self.branch_inputs, axis=0), dtype=torch.float32)
        self.trunk_inputs = torch.tensor(np.concatenate(self.trunk_inputs, axis=0), dtype=torch.float32)
        self.targets = torch.tensor(np.concatenate(self.targets, axis=0), dtype=torch.float32)
        
    def __len__(self):
        return self.branch_inputs.shape[0]
        
    def __getitem__(self, idx):
        return (self.branch_inputs[idx], self.trunk_inputs[idx]), self.targets[idx]
    
    def get_trajectory_by_idx(self, traj_idx):
        start = traj_idx * self.n_obs
        end = start + self.n_obs
        u_sample = self.branch_inputs[start:end]
        y_sample = self.trunk_inputs[start:end]
        target_sample = self.targets[start:end]
        return (u_sample, y_sample), target_sample
    
    
# --- Verification Script ---
if __name__ == "__main__":
    # 1. Dataset Initialization
    N_TRAJ = 5
    N_OBS = 50
    dataset = IdealMassSpringDataset(n_traj=N_TRAJ, n_obs=N_OBS)

    # 2. Check Shapes
    print("\n--- Shape Verification ---")
    print(f"Total Samples (N_TRAJ * N_OBS): {len(dataset)}")
    print(f"Expected: {N_TRAJ * N_OBS}")
    
    (u, y), target = dataset[0]
    print(f"Single Item Shapes:")
    print(f"  - Branch Input (u, Initial Condition): {u.shape} (Expected: torch.Size([2]))")
    print(f"  - Trunk Input (y, Time): {y.shape} (Expected: torch.Size([1]))")
    print(f"  - Target (State): {target.shape} (Expected: torch.Size([2]))")

    # 3. Check Data Consistency (Tiling Check)
    print("\n--- Consistency Verification (Trajectory 0) ---")
    (u_traj, y_traj), target_traj = dataset.get_trajectory_by_idx(0)
    
    # Check if 'u' (Initial Condition) is identical across all time steps in one trajectory
    u_std = torch.std(u_traj, dim=0)
    if torch.all(u_std < 1e-6):
        print("✅ PASS: Branch input is constant across the trajectory.")
    else:
        print(f"❌ FAIL: Branch input varies within trajectory. Std: {u_std}")

    # 4. Physics & Visualization Check
    print("\n--- Physics Verification (Visualization) ---")
    
    # Convert to numpy for plotting
    t_np = y_traj.numpy().flatten()
    q_np = target_traj[:, 0].numpy()
    p_np = target_traj[:, 1].numpy()
    
    # Calculate Energy (Hamiltonian)
    # H = 0.5 * k * q^2 + 0.5 * p^2 / m
    energy = 0.5 * dataset.k * q_np**2 + 0.5 * p_np**2 / dataset.mass
    energy_std = np.std(energy)
    
    print(f"Mean Energy: {np.mean(energy):.4f}")
    print(f"Energy Std Dev: {energy_std:.6f} (Should be close to 0)")

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Time vs State
    axes[0].plot(t_np, q_np, label='Position (q)', marker='.')
    axes[0].plot(t_np, p_np, label='Momentum (p)', marker='.')
    axes[0].set_title('Time Evolution')
    axes[0].set_xlabel('Time (t)')
    axes[0].set_ylabel('State')
    axes[0].legend()
    axes[0].grid(True)
    
    # Phase Space
    axes[1].plot(q_np, p_np, 'g-o')
    axes[1].set_title('Phase Space (q vs p)')
    axes[1].set_xlabel('Position (q)')
    axes[1].set_ylabel('Momentum (p)')
    axes[1].axis('equal')
    axes[1].grid(True)

    # Energy Conservation
    axes[2].plot(t_np, energy, 'r-')
    axes[2].set_title('Total Energy (H)')
    axes[2].set_xlabel('Time (t)')
    axes[2].set_ylabel('Energy')
    axes[2].set_ylim(np.mean(energy) - 0.1, np.mean(energy) + 0.1)
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("ideal_mass_spring_dataset_verification.pdf")