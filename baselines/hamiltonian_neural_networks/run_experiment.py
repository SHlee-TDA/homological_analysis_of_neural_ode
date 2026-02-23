import os

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from data import IdealMassSpringDataset
from model import HNN, BaselineNN
from trainer import Trainer 

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXPERIMENT_NAME = "Task1_IdealMassSpring"
    RESULTS_DIR = "./results"
    FIGURE_DIR = "./experiment_figures"
    
    N_TRAJ = 50
    N_OBS = 30
    NOISE_STD = 0.1
    T_SPAN = (0, 3) # Observation duration
    
    INPUT_DIM = 2
    HIDDEN_DIM = 200
    HIDDEN_LAYERS = 2
    ACTIVATION = 'tanh'
    
    # Training Params (논문 설정) 
    EPOCHS = 2000
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 750 # Full batch (25 * 30)
    WEIGHT_DECAY = 1e-4

def get_vector_field(model, x, is_hnn=True):
    x_tensor = torch.tensor(x, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
    
    if is_hnn:
        x_tensor.requires_grad_(True)
        # HNN: Symplectic Gradient 계산
        dxdt = model.symplectic_gradient(x_tensor)
    else:
        # Baseline: Forward pass
        dxdt = model(x_tensor)
        
    return dxdt.detach().cpu().numpy().flatten()

def integrate_model(model, t_span, y0, t_eval, is_hnn=True):
    def fun(t, np_x):
        return get_vector_field(model, np_x, is_hnn)
    
    # 논문과 동일하게 RK45, tolerance 1e-9 사용 [cite: 147]
    sol = solve_ivp(fun, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-9)
    return sol.y.T # (Time, 2)

def main():
    config = Config()
    
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    full_dataset = IdealMassSpringDataset(
        n_traj=config.N_TRAJ, 
        n_obs=config.N_OBS, 
        noise_std=config.NOISE_STD, 
        t_span=config.T_SPAN
    )
    
    train_size = 25 * config.N_OBS
    test_size = 25 * config.N_OBS
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"Dataset Split: Train {len(train_dataset.indices)} trajs, Test {len(test_dataset.indices)} trajs")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print("=== Training HNN ===")
    hnn_model = HNN(config.INPUT_DIM, config.HIDDEN_DIM, config.HIDDEN_LAYERS, config.ACTIVATION)
    hnn_trainer = Trainer(hnn_model, train_loader, test_loader, config)
    hnn_model, _ = hnn_trainer.train()
    
    print("\n=== Training Baseline NN ===")
    base_model = BaselineNN(config.INPUT_DIM, config.HIDDEN_DIM, config.HIDDEN_LAYERS, config.ACTIVATION)
    base_trainer = Trainer(base_model, train_loader, test_loader, config)
    base_model, _ = base_trainer.train()
    
    print("\n=== Generating Plots (Long-term Integration) ===")
    
    q0, p0 = full_dataset.sample_init() 
    
    t_eval = np.linspace(0, 20, 200) 
    
    true_sol = solve_ivp(
        lambda t, z: full_dataset.symplectic_gradient(z[0], z[1]), 
        (0, 20), [q0, p0], t_eval=t_eval, method='RK45', rtol=1e-9
    )
    true_y = true_sol.y.T

    hnn_y = integrate_model(hnn_model, (0, 20), [q0, p0], t_eval, is_hnn=True)
    base_y = integrate_model(base_model, (0, 20), [q0, p0], t_eval, is_hnn=False)
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # (1) Phase Space
    axes[0].plot(true_y[:,0], true_y[:,1], 'k-', label='Ground Truth', lw=2, alpha=0.3)
    axes[0].plot(base_y[:,0], base_y[:,1], 'r-', label='Baseline NN')
    axes[0].plot(hnn_y[:,0], hnn_y[:,1], 'b--', label='HNN')
    axes[0].set_title("Phase Space")
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("p")
    axes[0].legend()
    
    # (2) MSE
    mse_base = np.mean((base_y - true_y)**2, axis=1)
    mse_hnn = np.mean((hnn_y - true_y)**2, axis=1)
    
    axes[1].plot(t_eval, mse_base, 'r-', label='Baseline NN')
    axes[1].plot(t_eval, mse_hnn, 'b-', label='HNN')
    axes[1].set_title("MSE between coordinates")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    
    # (3) Total HNN-conserved quantity
    def calculate_hnn_energy(model, traj_np):
        """
        궤적 데이터(numpy)를 받아 학습된 HNN 모델의 에너지 함수값 H_theta(q, p)를 계산
        """
        tensor_traj = torch.tensor(traj_np, dtype=torch.float32, device=Config.DEVICE)
        with torch.no_grad():
            # HNN 모델의 forward는 H값을 반환함
            energies = model(tensor_traj).cpu().numpy().flatten()
        return energies

    # 3-1. Ground Truth 궤적을 HNN 에너지 함수에 대입 (검정)
    # 실제 물리 궤적은 에너지가 보존되므로, HNN이 잘 학습했다면 이 값도 상수여야 함
    hnn_E_on_true = calculate_hnn_energy(hnn_model, true_y)
    
    # 3-2. Baseline 궤적을 HNN 에너지 함수에 대입 (빨강)
    # Baseline 궤적은 엉망이므로 HNN 관점에서도 에너지가 요동쳐야 함
    hnn_E_on_base = calculate_hnn_energy(hnn_model, base_y)
    
    # 3-3. HNN 궤적을 HNN 에너지 함수에 대입 (파랑)
    # 자기 자신이 만든 궤적이므로 당연히 보존되어야 함
    hnn_E_on_hnn = calculate_hnn_energy(hnn_model, hnn_y)
    
    axes[2].plot(t_eval, hnn_E_on_true, 'k-', label='Ground Truth', lw=2, alpha=0.3)
    axes[2].plot(t_eval, hnn_E_on_base, 'r-', label='Baseline NN')
    axes[2].plot(t_eval, hnn_E_on_hnn, 'b--', label='HNN')
    axes[2].set_title("Total HNN-conserved quantity\n(Evaluated on H_theta)")
    axes[2].set_xlabel("Time Step")
    axes[2].legend()
    
    # (4) Total Energy
    # E = 0.5*k*q^2 + 0.5*p^2/m (k=1, m=1) [cite: 116]
    get_E = lambda traj: 0.5 * traj[:,0]**2 + 0.5 * traj[:,1]**2
    
    axes[3].plot(t_eval, get_E(true_y), 'k-', label='Ground Truth', lw=2, alpha=0.3)
    axes[3].plot(t_eval, get_E(base_y), 'r-', label='Baseline NN')
    axes[3].plot(t_eval, get_E(hnn_y), 'b--', label='HNN')
    axes[3].set_title("Total Energy Conservation")
    axes[3].set_xlabel("Time")
    axes[3].set_ylim(0, 1.0)
    axes[3].legend()
    
    plt.tight_layout()
    save_path = os.path.join(config.FIGURE_DIR, f"{config.EXPERIMENT_NAME}_results_4cols.pdf")
    plt.savefig(save_path)
    print(f"Figure saved successfully at: {save_path}")
    
if __name__ == "__main__":
    main()