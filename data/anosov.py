import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from typing import Tuple, Optional

class AnosoveDataset(Dataset):
    """
    Anosove diffeomorphism dataset.
    Each sample consists of a 1-step evolution pair (x_t, x_t+1) 
    """
    def __init__(self,
                 n_samples: int,
                 seed: Optional[int]=None):
        super().__init__()
        self.n_samples = n_samples
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # Sample initial states randomly
        self.initial_states = torch.rand(n_samples, 2, dtype=torch.float32)
        self.next_states = self.forward(self.initial_states)
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        f(x, y) = (2x + y, x + y) mod 1
        
        Args:
            states: torch tensor of shape (N, 2)
        
        Returns:
            next state  tensor of shape (N, 2)
        """
        x, y = states[:, 0], states[:, 1]
        
        x_next = (2*x + y) % 1.0
        y_next = (x + y) % 1.0
        return torch.stack([x_next, y_next], dim=1)
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.initial_states[idx], self.next_states[idx]
    

def create_dataloaders(
    n_train: int,
    n_test: int,
    batch_size: int,
    seed: Optional[int]=None,
    num_workers: int=0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test loader
    """
    
    train_dataset = AnosoveDataset(
        n_samples=n_train,
        seed=seed
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    test_seed = seed + 1 if seed is not None else None
    test_dataset = AnosoveDataset(
        n_samples=n_test,
        seed=test_seed
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def generate_grid_pattern(n_points: int = 500) -> torch.Tensor:
    """
    시각화용 균일 그리드 패턴 생성
    
    Args:
        n_points: 총 점 개수 (가장 가까운 제곱수로 조정됨)
    
    Returns:
        (N, 2) 토러스 상의 점군
    """
    n_side = int(np.sqrt(n_points))
    x = torch.linspace(0, 1, n_side + 1)[:-1]  # [0, 1) 범위
    y = torch.linspace(0, 1, n_side + 1)[:-1]
    
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    return points


def generate_cat_face_pattern(n_points: int = 500) -> torch.Tensor:
    """
    시각화용 '고양이 얼굴' 패턴 생성
    
    간단한 기하학적 패턴으로 대체:
    - 중심 원 (얼굴)
    - 두 개의 작은 원 (눈)
    - 삼각형 (귀)
    
    Args:
        n_points: 총 점 개수
    
    Returns:
        (N, 2) 토러스 상의 점군
    """
    points_list = []
    
    # 얼굴 (큰 원)
    n_face = n_points // 2
    theta = torch.linspace(0, 2 * np.pi, n_face)
    r = 0.25
    x_face = 0.5 + r * torch.cos(theta)
    y_face = 0.5 + r * torch.sin(theta)
    points_list.append(torch.stack([x_face, y_face], dim=1))
    
    # 왼쪽 눈 (작은 원)
    n_eye = n_points // 6
    theta = torch.linspace(0, 2 * np.pi, n_eye)
    r = 0.05
    x_left_eye = 0.4 + r * torch.cos(theta)
    y_left_eye = 0.55 + r * torch.sin(theta)
    points_list.append(torch.stack([x_left_eye, y_left_eye], dim=1))
    
    # 오른쪽 눈
    x_right_eye = 0.6 + r * torch.cos(theta)
    y_right_eye = 0.55 + r * torch.sin(theta)
    points_list.append(torch.stack([x_right_eye, y_right_eye], dim=1))
    
    # 귀 (삼각형 라인)
    n_ear = n_points // 6
    # 왼쪽 귀
    x_left_ear = torch.linspace(0.35, 0.4, n_ear // 2)
    y_left_ear = torch.linspace(0.7, 0.8, n_ear // 2)
    points_list.append(torch.stack([x_left_ear, y_left_ear], dim=1))
    
    # 오른쪽 귀
    x_right_ear = torch.linspace(0.6, 0.65, n_ear // 2)
    y_right_ear = torch.linspace(0.7, 0.8, n_ear // 2)
    points_list.append(torch.stack([x_right_ear, y_right_ear], dim=1))
    
    # 모든 점 결합
    points = torch.cat(points_list, dim=0)
    
    # [0, 1) 범위로 정규화
    points = points % 1.0
    
    return points