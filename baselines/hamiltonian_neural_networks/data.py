import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset


class IdealMassSpringDataset(Dataset):
    def __init__(self, 
                 mass=1.0, k=1.0, 
                 n_traj= 50, n_obs=30, 
                 noise_std=0.1, 
                 t_span=(0, 3)
                 ):
        self.mass = mass
        self.k = k
        self.n_traj = n_traj
        self.n_obs = n_obs
        self.noise_std = noise_std
        self.t_span = t_span
        self.X, self.y = self._create_dataset()
        
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
        q0, p0 = self.sample_init()
        t0, t1 = self.t_span
        t_eval = np.linspace(t0, t1, self.n_obs)
        dydt = lambda t, z: self.symplectic_gradient(z[0], z[1])
        
        sol = solve_ivp(
            fun=dydt,
            t_span=self.t_span,
            y0=[q0, p0],
            method='RK45',
            t_eval=t_eval,
            rtol=1e-10
        )

        q, p = sol['y']
        dqdt, dpdt = self.symplectic_gradient(q, p)
        
        q += np.random.normal(0, self.noise_std, size=q.shape)
        p += np.random.normal(0, self.noise_std, size=p.shape)

        X = np.stack([q, p], axis=1)    # (time_steps, 2)
        y = np.stack([dqdt, dpdt], axis=1)  # (time_steps, 2)
        return X, y
    
    def _create_dataset(self):
        data_X = []
        data_y = []
        for _ in range(self.n_traj):
            X, y = self.generate_trajectory()
            data_X.append(X)
            data_y.append(y)
        data_X = np.concatenate(data_X, axis=0)    # (n_obs*time_steps, 2)
        data_y = np.concatenate(data_y, axis=0)    # (n_obs*time_steps, 2)
        
        X_tensor = torch.tensor(data_X, dtype=torch.float32)
        y_tensor = torch.tensor(data_y, dtype=torch.float32)
        return X_tensor, y_tensor
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]        

class HenonHeliesDataset(Dataset):
    def __init__(self, 
                 mass=1.0, k=1.0, 
                 n_traj= 50, n_obs=30, 
                 noise_std=0.1, 
                 t_span=(0, 3)
                 ):
        self.mass = mass
        self.k = k
        self.n_traj = n_traj
        self.n_obs = n_obs
        self.noise_std = noise_std
        self.t_span = t_span
        self.X, self.y = self._create_dataset()
        
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
        q0, p0 = self.sample_init()
        t0, t1 = self.t_span
        t_eval = np.linspace(t0, t1, self.n_obs)
        dydt = lambda t, z: self.symplectic_gradient(z[0], z[1])
        
        sol = solve_ivp(
            fun=dydt,
            t_span=self.t_span,
            y0=[q0, p0],
            method='RK45',
            t_eval=t_eval,
            rtol=1e-10
        )

        q, p = sol['y']
        dqdt, dpdt = self.symplectic_gradient(q, p)
        
        q += np.random.normal(0, self.noise_std, size=q.shape)
        p += np.random.normal(0, self.noise_std, size=p.shape)

        X = np.stack([q, p], axis=1)    # (time_steps, 2)
        y = np.stack([dqdt, dpdt], axis=1)  # (time_steps, 2)
        return X, y
    
    def _create_dataset(self):
        data_X = []
        data_y = []
        for _ in range(self.n_traj):
            X, y = self.generate_trajectory()
            data_X.append(X)
            data_y.append(y)
        data_X = np.concatenate(data_X, axis=0)    # (n_obs*time_steps, 2)
        data_y = np.concatenate(data_y, axis=0)    # (n_obs*time_steps, 2)
        
        X_tensor = torch.tensor(data_X, dtype=torch.float32)
        y_tensor = torch.tensor(data_y, dtype=torch.float32)
        return X_tensor, y_tensor
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  

if __name__ == "__main__":
    dataset = IdealMassSpringDataset()
    print("Dataset size:", len(dataset))
    sample_X, sample_y = dataset[0]
    print("Sample X:", sample_X)
    print("Sample y:", sample_y)