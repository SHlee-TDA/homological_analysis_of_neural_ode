import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch.utils.data import Dataset

class EnergyBandDataset(Dataset):
    def __init__(self, energy_range, mass=1.0, k=1.0, n_traj=50, n_obs=100, t_span=(0, 10)):
        self.energy_range = energy_range # (E_min, E_max)
        self.mass = mass
        self.k = k
        self.n_traj = n_traj
        self.n_obs = n_obs
        self.t_span = t_span
        
        self.data_deeponet = [] # (u, y, target)
        self.data_hnn = []      # (state, dstate_dt)
        
        self._create_dataset()

    def sample_init_from_energy(self):
        # 지정된 에너지 밴드 내에서 균등 샘플링
        E = np.random.uniform(self.energy_range[0], self.energy_range[1])
        theta = np.random.uniform(0, 2*np.pi)
        
        # H = p^2/2m + 1/2kq^2 = E
        q0 = np.sqrt(2*E/self.k) * np.cos(theta)
        p0 = np.sqrt(2*self.mass*E) * np.sin(theta)
        return np.array([q0, p0])
    
    def symplectic_gradient(self, q, p):
        return np.array([p/self.mass, -self.k*q])

    def _create_dataset(self):
        t_eval = np.linspace(self.t_span[0], self.t_span[1], self.n_obs)
        
        for _ in range(self.n_traj):
            z0 = self.sample_init_from_energy()
            
            # Solve Trajectory
            sol = solve_ivp(lambda t, z: self.symplectic_gradient(z[0], z[1]), 
                            self.t_span, z0, t_eval=t_eval, rtol=1e-9)
            states = sol.y.T # (Time, 2)
            
            # --- For DeepONet ---
            # u: 초기값 반복 (Time, 2), y: 시간 (Time, 1)
            u = np.tile(z0, (self.n_obs, 1))
            y = t_eval.reshape(-1, 1)
            self.data_deeponet.append(((u, y), states))
            
            # --- For HNN/Baseline ---
            # Input: state(t), Target: gradient(t)
            grads = np.array([self.symplectic_gradient(s[0], s[1]) for s in states])
            self.data_hnn.append((states, grads))

    def __len__(self):
        return self.n_traj * self.n_obs

    def __getitem__(self, idx):
        # 편의상 DeepONet용과 HNN용 데이터를 분리해서 접근하는 인터페이스는 
        # DataLoader 생성 시 collate_fn이나 별도 처리가 필요하지만,
        # 여기서는 간단히 DeepONet 포맷을 기본으로 하되 get_hnn_data 메서드를 둡니다.
        traj_idx = idx // self.n_obs
        time_idx = idx % self.n_obs
        
        (u_full, y_full), target_full = self.data_deeponet[traj_idx]
        
        u = torch.tensor(u_full[time_idx], dtype=torch.float32)
        y = torch.tensor(y_full[time_idx], dtype=torch.float32)
        target = torch.tensor(target_full[time_idx], dtype=torch.float32)
        
        return (u, y), target

    def get_hnn_dataset(self):
        # HNN 학습용 평탄화된 Tensor 반환
        all_states = []
        all_grads = []
        for s, g in self.data_hnn:
            all_states.append(s)
            all_grads.append(g)
        
        X = torch.tensor(np.concatenate(all_states, axis=0), dtype=torch.float32)
        y = torch.tensor(np.concatenate(all_grads, axis=0), dtype=torch.float32)
        
        # TensorDataset으로 포장해서 반환
        return torch.utils.data.TensorDataset(X, y)