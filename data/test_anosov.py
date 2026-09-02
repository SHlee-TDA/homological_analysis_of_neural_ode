import pytest
import torch
from anosov import AnosoveDataset, create_dataloaders, generate_grid_pattern, generate_cat_face_pattern

import matplotlib.pyplot as plt


class TestAnosoveDataset:
    def test_dataset_initialization(self):
        """Test basic dataset initialization"""
        n_samples = 100
        dataset = AnosoveDataset(n_samples=n_samples, seed=42)
        
        assert dataset.n_samples == n_samples
        assert dataset.initial_states.shape == (n_samples, 2)
        assert dataset.next_states.shape == (n_samples, 2)
        assert torch.all(dataset.initial_states >= 0) and torch.all(dataset.initial_states < 1)
        assert torch.all(dataset.next_states >= 0) and torch.all(dataset.next_states < 1)
    
    def test_dataset_reproducibility(self):
        """Test that same seed produces same data"""
        dataset1 = AnosoveDataset(n_samples=50, seed=123)
        dataset2 = AnosoveDataset(n_samples=50, seed=123)
        
        assert torch.allclose(dataset1.initial_states, dataset2.initial_states)
        assert torch.allclose(dataset1.next_states, dataset2.next_states)
    
    def test_getitem(self):
        """Test __getitem__ method"""
        dataset = AnosoveDataset(n_samples=10, seed=42)
        
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (2,)
        assert y.shape == (2,)
    
    def test_forward_transformation(self):
        """Test forward transformation f(x,y) = (2x+y, x+y) mod 1"""
        dataset = AnosoveDataset(n_samples=5, seed=42)
        
        test_states = torch.tensor([[0.3, 0.2], [0.5, 0.5]], dtype=torch.float32)
        result = dataset.forward(test_states)
        
        # Manual calculation: f(0.3, 0.2) = (2*0.3 + 0.2, 0.3 + 0.2) mod 1 = (0.8, 0.5)
        expected_0 = torch.tensor([0.8, 0.5], dtype=torch.float32)
        # Manual calculation: f(0.5, 0.5) = (2*0.5 + 0.5, 0.5 + 0.5) mod 1 = (0.5, 0.0)
        expected_1 = torch.tensor([0.5, 0.0], dtype=torch.float32)
        
        assert torch.allclose(result[0], expected_0, atol=1e-6)
        assert torch.allclose(result[1], expected_1, atol=1e-6)


class TestDataLoaders:
    def test_create_dataloaders(self):
        """Test dataloader creation"""
        train_loader, test_loader = create_dataloaders(
            n_train=100,
            n_test=20,
            batch_size=16,
            seed=42
        )
        
        assert train_loader is not None
        assert test_loader is not None
        
        # Check batch size
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] == 16
        assert batch_y.shape[0] == 16


class TestPatternGeneration:
    def test_grid_pattern(self):
        """Test grid pattern generation"""
        points = generate_grid_pattern(n_points=100)
        
        assert points.shape[0] <= 100
        assert points.shape[1] == 2
        assert torch.all(points >= 0) and torch.all(points < 1)
    
    def test_cat_face_pattern(self):
        """Test cat face pattern generation"""
        points = generate_cat_face_pattern(n_points=100)
        
        assert points.shape[1] == 2
        assert torch.all(points >= 0) and torch.all(points < 1)


class TestVisualization:
    def test_visualize_patterns_evolution(self):
        """Visualize grid and cat face patterns under Anosov evolution"""
        dataset = AnosoveDataset(n_samples=1, seed=42)
        
        grid_points = generate_grid_pattern(n_points=400)
        cat_points = generate_cat_face_pattern(n_points=400)
        
        grid_evolved = grid_points.clone()
        cat_evolved = cat_points.clone()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Grid evolution
        for i in range(3):
            ax = axes[0, i]
            ax.scatter(grid_evolved[:, 0], grid_evolved[:, 1], s=1, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f'Grid Pattern - Step {i}')
            grid_evolved = dataset.forward(grid_evolved)
        
        # Cat face evolution
        cat_evolved = cat_points.clone()
        for i in range(3):
            ax = axes[1, i]
            ax.scatter(cat_evolved[:, 0], cat_evolved[:, 1], s=1, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f'Cat Face Pattern - Step {i}')
            cat_evolved = dataset.forward(cat_evolved)
        
        # Tracking a trajectory
        x0 = torch.tensor([0.1, 0.1])
        trajectory = [x0]
        x_curr = x0
        for i in range(1000):
            x_next = dataset.forward(x_curr.unsqueeze(0))[0]
            trajectory.append(x_next)
            x_curr = x_next
        
        # trajectory를 텐서로 변환
        trajectory = torch.stack(trajectory)
        
        # 여러 시각화 방법
        fig_traj, axes_traj = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 색상이 칠해진 line plot (시간 진행 표시)
        ax = axes_traj[0, 0]
        steps = range(len(trajectory))
        scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], c=steps, s=20, cmap='viridis', alpha=0.7)
        ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3, linewidth=0.5, color='gray')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Trajectory with Time Coloring')
        ax.legend()
        cbar1 = plt.colorbar(scatter, ax=ax)
        cbar1.set_label('Step')
        
        # 2. x, y 좌표 시간 변화
        ax = axes_traj[0, 1]
        ax.plot(steps, trajectory[:, 0], label='x(t)', linewidth=2)
        ax.plot(steps, trajectory[:, 1], label='y(t)', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Coordinate Value')
        ax.set_title('Coordinate Values over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 연속된 점들 연결 (화살표로 진행 방향 표시)
        ax = axes_traj[1, 0]
        for i in range(0, len(trajectory)-1, 5):  # 5개 간격으로 화살표 표시
            ax.arrow(trajectory[i, 0], trajectory[i, 1],
                    trajectory[i+1, 0] - trajectory[i, 0],
                    trajectory[i+1, 1] - trajectory[i, 1],
                    head_width=0.02, head_length=0.02, fc='blue', ec='blue', alpha=0.5)
        ax.scatter(trajectory[:, 0], trajectory[:, 1], c=steps, s=10, cmap='viridis', alpha=0.5)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Trajectory with Direction Arrows')
        ax.legend()
        
        # 4. Lyapunov exponent 추정 (인접한 점들 사이의 거리)
        ax = axes_traj[1, 1]
        distances = torch.norm(trajectory[1:] - trajectory[:-1], dim=1)
        ax.semilogy(steps[1:], distances, linewidth=2, color='purple')
        ax.set_xlabel('Step')
        ax.set_ylabel('Distance to Next Point (log scale)')
        ax.set_title('Local Expansion Rate')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig('anosov_trajectory_evolution.png', dpi=100)
        plt.close(fig_traj)
        
        plt.tight_layout()
        plt.savefig('anosov_patterns_evolution.png', dpi=100)
        plt.close(fig)  # 명시적으로 fig 해제
        
        # 검증: trajectory 길이와 데이터 무결성 확인
        assert len(trajectory) == 1001  # 초기 상태 + 1000 steps
        assert trajectory.shape[1] == 2
        assert torch.all(trajectory >= 0) and torch.all(trajectory < 1)