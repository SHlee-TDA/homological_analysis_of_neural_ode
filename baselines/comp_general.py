import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.integrate import solve_ivp


from data import EnergyBandDataset
from deeponet.model import SimpleDeepONet
from deeponet.trainer import Trainer as DeepONetTrainer
from hamiltonian_neural_networks.model import HNN, BaselineNN
from hamiltonian_neural_networks.trainer import Trainer as HNNTrainer


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXPERIMENT_NAME = "Energy_Generalization_Test"
    RESULTS_DIR = "./results"
    
    # Energy Bands
    TRAIN_ENERGY = (0.2, 0.6)  # 학습 영역 (Low Energy)
    TEST_ENERGY = (1.0, 1.4)   # 테스트 영역 (High Energy, OOD)
    
    N_TRAJ = 30
    N_OBS = 100
    EPOCHS = 1000
    LEARNING_RATE = 1e-3

def get_hnn_trajectory(model, z0, t_span, t_eval):
    """HNN/Baseline용 적분기"""
    def func(t, x):
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
        if hasattr(model, 'symplectic_gradient'): # HNN
            grad = model.symplectic_gradient(x_tensor)
        else: # Baseline
            grad = model(x_tensor)
        return grad.detach().cpu().numpy().flatten()
    
    sol = solve_ivp(func, t_span, z0, t_eval=t_eval, method='RK45', rtol=1e-9)
    return sol.y.T

def main():
    # 1. 데이터셋 생성
    print("Generating Datasets...")
    # Train: Low Energy
    train_ds_obj = EnergyBandDataset(Config.TRAIN_ENERGY, n_traj=Config.N_TRAJ, n_obs=Config.N_OBS)
    
    # DeepONet Loaders
    don_train_loader = DataLoader(train_ds_obj, batch_size=1024, shuffle=True)
    
    # HNN Loaders
    hnn_train_ds = train_ds_obj.get_hnn_dataset()
    hnn_train_loader = DataLoader(hnn_train_ds, batch_size=1024, shuffle=True)

    # 2. 모델 학습
    print("\n=== Training DeepONet ===")
    don_model = SimpleDeepONet(branch_input_dim=2, trunk_input_dim=1, latent_dim=64, hidden_width=128)
    don_trainer = DeepONetTrainer(don_model, don_train_loader, don_train_loader, Config) # Val=Train for simplicity
    don_model, _ = don_trainer.train()

    print("\n=== Training HNN ===")
    hnn_model = HNN(input_dim=2, hidden_dim=128, hidden_layers=2, activation='tanh')
    hnn_trainer = HNNTrainer(hnn_model, hnn_train_loader, hnn_train_loader, Config)
    hnn_model, _ = hnn_trainer.train()

    print("\n=== Training Baseline ===")
    base_model = BaselineNN(input_dim=2, hidden_dim=128, hidden_layers=2, activation='tanh')
    base_trainer = HNNTrainer(base_model, hnn_train_loader, hnn_train_loader, Config)
    base_model, _ = base_trainer.train()

    # 3. 시각화 및 평가 (Phase Space Contour)
    print("\n=== Evaluating & Plotting ===")
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    # Background Hamiltonian Energy Grid
    x = np.linspace(-3.5, 3.5, 200)
    p = np.linspace(-3.5, 3.5, 200)
    X, P = np.meshgrid(x, p)
    H = 0.5 * 1.0 * X**2 + 0.5 * P**2 / 1.0 # k=1, m=1
    
    # RGBA 색상 정의 (투명도 설정)
    train_color_fill = (0.0, 0.0, 1.0, 0.15) # 파란색 채우기 (옅게)
    test_color_fill = (1.0, 0.0, 0.0, 0.15)  # 빨간색 채우기 (옅게)

    for i in range(2): 
        # [수정] 영역 채우기 (contourf 이용)
        # levels=[min, max]로 주면 [미만, 사이, 초과] 3개 영역이 생김.
        # 가운데(사이) 영역에만 색을 할당하고 나머지는 투명(None) 처리
        
        # Train Band 채우기 (파랑)
        ax[i].contourf(X, P, H, levels=[Config.TRAIN_ENERGY[0], Config.TRAIN_ENERGY[1]],
                       colors=[(0,0,0,0), train_color_fill, (0,0,0,0)])
        
        # Test Band 채우기 (빨강)
        ax[i].contourf(X, P, H, levels=[Config.TEST_ENERGY[0], Config.TEST_ENERGY[1]],
                       colors=[(0,0,0,0), test_color_fill, (0,0,0,0)])
        
        # [수정] 경계선 그리기 (contour 이용) - 더 선명하게
        ax[i].contour(X, P, H, levels=Config.TRAIN_ENERGY, colors=['blue'], linewidths=1.5, linestyles='-')
        ax[i].contour(X, P, H, levels=Config.TEST_ENERGY, colors=['red'], linewidths=1.5, linestyles='-')
        
        # 축 및 레이블 설정
        ax[i].set_xlabel("Position (q)", fontsize=12)
        ax[i].set_ylabel("Momentum (p)", fontsize=12)
        ax[i].set_aspect('equal')
        ax[i].grid(True, which='both', linestyle=':', linewidth=0.5)
        ax[i].set_xlim([-3.5, 3.5])
        ax[i].set_ylim([-3.5, 3.5])

    ax[0].set_title("In-Distribution (Train Band)", fontsize=14, fontweight='bold', color='blue')
    ax[1].set_title("Out-of-Distribution (Test Band)", fontsize=14, fontweight='bold', color='red')

    # 궤적 비교
    t_eval = np.linspace(0, 15, 400) # 테스트 시간도 충분히 길게 설정
    
    # ID Sample (Train Band 중간값)
    E_id = np.mean(Config.TRAIN_ENERGY)
    z0_in = np.array([np.sqrt(2*E_id), 0.0])

    # OOD Sample (Test Band 중간값)
    E_ood = np.mean(Config.TEST_ENERGY)
    z0_out = np.array([np.sqrt(2*E_ood), 0.0])

    # Plot Loop
    for z0, axis in [(z0_in, ax[0]), (z0_out, ax[1])]:
        # Ground Truth
        true_sol = solve_ivp(lambda t, z: [z[1], -z[0]], (0, 15), z0, t_eval=t_eval, rtol=1e-10)
        true_y = true_sol.y.T
        axis.plot(true_y[:,0], true_y[:,1], 'k-', lw=4, alpha=0.2, label='Ground Truth')
        
        # HNN
        hnn_y = get_hnn_trajectory(hnn_model, z0, (0, 15), t_eval)
        axis.plot(hnn_y[:,0], hnn_y[:,1], 'g--', lw=2, label='HNN')

        # Baseline
        base_y = get_hnn_trajectory(base_model, z0, (0, 15), t_eval)
        axis.plot(base_y[:,0], base_y[:,1], color='orange', linestyle=':', lw=2, label='Baseline')

        # DeepONet
        don_model.eval()
        with torch.no_grad():
            u_in = torch.tensor(z0, dtype=torch.float32).unsqueeze(0).repeat(len(t_eval), 1).to(Config.DEVICE)
            y_in = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1).to(Config.DEVICE)
            don_y = don_model(u_in, y_in).cpu().numpy()
            
        axis.plot(don_y[:,0], don_y[:,1], color='magenta', linestyle='-.', lw=2.5, label='DeepONet')
        
        # 시작점
        axis.scatter(z0[0], z0[1], c='black', s=150, marker='*', zorder=10)

    # 범례는 하나만 표시
    ax[1].legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig("./energy_generalization_test.png")
    plt.show()

if __name__ == "__main__":
    main()