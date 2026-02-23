import os
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from data import IdealMassSpringDataset  # (앞서 작성한 데이터셋 클래스 파일명)
from model import SimpleDeepONet            # (앞서 작성한 모델 클래스 파일명)
from trainer import Trainer # (앞서 작성한 Trainer 및 Config)

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXPERIMENT_NAME = "Task1_IdealMassSpring"
    RESULTS_DIR = "./results"
    FIGURE_DIR = "./experiment_figures"
    
    N_TRAJ = 50
    N_OBS = 30
    NOISE_STD = 0.0
    T_SPAN = (0, 3) # Observation duration
    
    # 모델 파라미터
    BRANCH_DIM = 2
    TRUNK_DIM = 1
    HIDDEN_WIDTH = 128
    HIDDEN_DEPTH = 3
    LATENT_DIM = 64
    
    # 학습 파라미터 
    EPOCHS = 2000
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 750 
    WEIGHT_DECAY = 1e-4

def predict_trajectory_deeponet(model, z0, t_eval):
    """
    DeepONet을 사용하여 궤적 예측 (Solver 없이 직접 예측)
    z0: (2,) numpy array
    t_eval: (Time,) numpy array
    """
    model.eval()
    with torch.no_grad():
        # 1. 입력 생성 (Batch 처리를 위해 확장)
        # u: 초기 조건 (Time, 2) - 모든 시간 스텝에 대해 동일
        u_tensor = torch.tensor(z0, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0).repeat(len(t_eval), 1)
        
        # y: 시간 (Time, 1)
        y_tensor = torch.tensor(t_eval, dtype=torch.float32, device=Config.DEVICE).unsqueeze(1)
        
        # 2. Forward
        pred_tensor = model(u_tensor, y_tensor) # (Time, 2)
        
    return pred_tensor.cpu().numpy()

def main():
    config = Config()
    
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1. 데이터셋 준비
    full_dataset = IdealMassSpringDataset(
        n_traj=config.N_TRAJ, 
        n_obs=config.N_OBS, 
        noise_std=config.NOISE_STD, 
        t_span=config.T_SPAN
    )
    
    # Train/Test Split (궤적 단위로 나누는 것이 아님에 주의 - 현재 데이터셋은 전체 섞여있음)
    # 엄밀한 테스트를 위해서는 '보지 못한 초기 조건'에 대한 일반화를 테스트해야 하므로,
    # Dataset 클래스에서 Trajectory 단위로 Split을 지원하거나, 
    # 여기서는 간단히 전체 데이터를 섞어서 Split 합니다.
    
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"Dataset Split: Train {len(train_dataset)} samples, Test {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 2. 모델 학습
    print("=== Training DeepONet ===")
    model = SimpleDeepONet(
        branch_input_dim=config.BRANCH_DIM,
        trunk_input_dim=config.TRUNK_DIM,
        hidden_depth=config.HIDDEN_DEPTH,
        hidden_width=config.HIDDEN_WIDTH,
        latent_dim=config.LATENT_DIM
    )
    
    trainer = Trainer(model, train_loader, test_loader, config)
    model, history = trainer.train()
    
    # 3. 결과 분석 (Long-term Prediction)
    print("\n=== Generating Plots (Long-term Extrapolation) ===")
    
    # 3-1. 테스트용 초기 조건 하나 샘플링 (학습 데이터에 없는 새로운 조건)
    z0 = full_dataset.sample_init() 
    
    # 3-2. 긴 시간(Extrapolation)에 대해 Ground Truth 생성
    # 학습은 T=3까지 했지만, 테스트는 T=10까지 수행하여 일반화 능력 검증
    t_eval = np.linspace(0, 10, 200) 
    
    true_sol = solve_ivp(
        lambda t, z: full_dataset.symplectic_gradient(z[0], z[1]), # symplectic_gradient는 인자 수정 필요할 수 있음
        (0, 10), z0, t_eval=t_eval, method='RK45', rtol=1e-9
    )
    true_y = true_sol.y.T

    # 3-3. DeepONet 예측
    pred_y = predict_trajectory_deeponet(model, z0, t_eval)
    
    # 4. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # (1) Phase Space
    axes[0].plot(true_y[:,0], true_y[:,1], 'k-', label='Ground Truth', lw=2, alpha=0.3)
    axes[0].plot(pred_y[:,0], pred_y[:,1], 'b--', label='DeepONet', lw=2)
    axes[0].set_title("Phase Space Trajectory")
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("p")
    axes[0].legend()
    
    # (2) MSE over Time
    mse = np.mean((pred_y - true_y)**2, axis=1)
    axes[1].plot(t_eval, mse, 'r-', label='MSE')
    axes[1].axvline(x=config.T_SPAN[1], color='g', linestyle=':', label='Train Domain End') # 학습 영역 표시
    axes[1].set_title("MSE over Time")
    axes[1].set_xlabel("Time (t)")
    axes[1].set_ylabel("MSE")
    axes[1].set_yscale('log')
    axes[1].legend()
    
    # (3) Total Energy Conservation
    # H = 0.5*k*q^2 + 0.5*p^2/m
    k, m = full_dataset.k, full_dataset.mass
    get_E = lambda traj: 0.5 * k * traj[:,0]**2 + 0.5 * (traj[:,1]**2) / m
    
    true_E = get_E(true_y)
    pred_E = get_E(pred_y)
    
    axes[2].plot(t_eval, true_E, 'k-', label='Ground Truth', lw=2, alpha=0.3)
    axes[2].plot(t_eval, pred_E, 'b--', label='DeepONet')
    axes[2].axvline(x=config.T_SPAN[1], color='g', linestyle=':', label='Train Domain End')
    axes[2].set_title("Total Energy Conservation")
    axes[2].set_xlabel("Time (t)")
    axes[2].set_ylabel("Total Energy (H)")
    axes[2].legend()
    
    plt.tight_layout()
    save_path = os.path.join(config.FIGURE_DIR, f"{config.EXPERIMENT_NAME}_deeponet_results.pdf")
    plt.savefig(save_path)
    print(f"Figure saved successfully at: {save_path}")

if __name__ == "__main__":
    main()