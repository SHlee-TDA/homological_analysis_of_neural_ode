import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_suspension_1d(ax, map_func, title, steps=100, n_orbits=5):
    # 시간축 t를 각도 theta로 변환 (0 ~ 2pi)
    # 시각화를 위해 연속적인 흐름처럼 보이게 선형 보간(Linear Interpolation)을 사용합니다.
    # 엄밀한 Suspension은 수직으로 올라가서 점프하는 것이지만, 
    # 위상수학적으로는 이를 이어 붙인 것이므로 대각선으로 흐르는 것과 동치입니다.
    
    t = np.linspace(0, 2*np.pi, 100) # 한 바퀴(t=0 to 1)를 각도로 표현
    
    # 여러 개의 궤적을 그림
    start_points = np.linspace(0.1, 0.9, n_orbits)
    
    for x0 in start_points:
        x_curr = x0
        full_x = []
        full_y = []
        full_z = []
        
        for _ in range(steps): # 몇 바퀴 돌 것인가
            x_next = map_func(x_curr)
            
            # 현재 x에서 다음 x로 부드럽게 이동하는 궤적 생성 (Suspension Flow)
            # t가 0->1 (angle 0->2pi)로 갈 때 x도 x_curr -> x_next로 이동
            x_interpolated = np.linspace(x_curr, x_next, len(t))
            
            # Cylinder 좌표계 변환
            # Radius = 2 + x (1차원 공간을 반지름 방향으로 띄움)
            r = 2 + x_interpolated
            theta = t # 시간은 각도로
            
            # 3D 좌표
            X = r * np.cos(theta)
            Y = r * np.sin(theta)
            Z = np.linspace(_, _+1, len(t)) * 0.5 # z축은 전체 시간의 흐름 (나선형으로 보기 위해)
            
            # 만약 위상학적 구조(Torus) 자체를 보고 싶다면 Z를 0으로 두면 되지만,
            # 궤적의 얽힘을 보기 위해 Z를 증가시켜 코일처럼 그립니다.
            
            ax.plot(X, Y, Z, alpha=0.7, linewidth=1)
            x_curr = x_next

    ax.set_title(title)
    ax.set_axis_off()

# 맵 정의
def simple_rotation(x, w=0.1):
    return (x + w) % 1.0

def logistic_map(x, r=3.9):
    return r * x * (1 - x)

# 시각화 실행
fig = plt.subplots(figsize=(12, 6))
ax1 = plt.subplot(121, projection='3d')
plot_suspension_1d(ax1, lambda x: simple_rotation(x, 0.05), "Suspension of Rotation (Torus Flow)", steps=20)

ax2 = plt.subplot(122, projection='3d')
plot_suspension_1d(ax2, lambda x: logistic_map(x, 3.9), "Suspension of Logistic Map (Stretch & Fold)", steps=20)

plt.tight_layout()
plt.savefig("suspension_flows.pdf")


def plot_suspension_2d_box(ax, map_func, title, n_points=500):
    # 초기 점 생성 (그림이나 격자)
    x = np.linspace(0.1, 0.9, 10)
    y = np.linspace(0.1, 0.9, 10)
    X0, Y0 = np.meshgrid(x, y)
    points = np.stack([X0.ravel(), Y0.ravel()], axis=1)
    
    # 색상 (초기 위치 기억)
    colors = points[:, 0] # x좌표에 따라 색상 부여
    
    # 궤적 계산 (t=0 -> t=1)
    # 여기서는 "흐름"을 보여주기 위해 선형 보간을 하되, 
    # 맵의 특성(자르고 붙이기)을 고려해야 함.
    # 단순히 선을 이으면 '자르는' 불연속성이 시각화를 망침.
    # 따라서, 점들이 t=0에서 t=1로 이동하는 '벡터장' 느낌으로 그림.
    
    # t=0 (바닥)
    ax.scatter(points[:,0], points[:,1], 0, c=colors, cmap='viridis', s=5, alpha=0.5, label='t=0')
    
    # t=1 (천장 - 맵 적용 후)
    next_points = map_func(points)
    ax.scatter(next_points[:,0], next_points[:,1], 1, c=colors, cmap='viridis', s=10, marker='^', label='t=1 (Mapped)')
    
    # 흐름선 그리기 (Flow lines)
    # Baker map처럼 불연속적인 경우, 중간 경로가 물리적으로 찢어져야 함.
    # 시각적 이해를 위해 각 점을 잇는 선을 그립니다.
    for i in range(len(points)):
        # 불연속점이 아닌 경우에만 선을 그림 (시각적 깔끔함을 위해 거리 체크)
        dist = np.linalg.norm(points[i] - next_points[i])
        # Arnold Cat map같은 Torus map은 경계 넘어가면 선 그리면 안됨
        if dist < 0.8: 
            ax.plot([points[i,0], next_points[i,0]], 
                    [points[i,1], next_points[i,1]], 
                    [0, 1], c='gray', alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time (t)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)

# 맵 정의
def bakers_map_2d(p):
    x, y = p[:, 0], p[:, 1]
    # 교과서적인 Baker Map (Standard)
    # x -> 2x mod 1
    # y -> y/2 (if x<0.5) or (y+1)/2 (if x>=0.5)
    
    mask = x < 0.5
    new_x = np.where(mask, 2*x, 2*x - 1)
    new_y = np.where(mask, 0.5*y, 0.5*y + 0.5)
    
    return np.stack([new_x, new_y], axis=1)

def arnold_cat_map(p):
    x, y = p[:, 0], p[:, 1]
    # [ 2 1 ]
    # [ 1 1 ]
    new_x = (2*x + y) % 1.0
    new_y = (x + y) % 1.0
    return np.stack([new_x, new_y], axis=1)

# 시각화 실행
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
plot_suspension_2d_box(ax1, bakers_map_2d, "Suspension of Baker's Map\n(Hyperbolic 3-Manifold Structure)")

ax2 = fig.add_subplot(122, projection='3d')
plot_suspension_2d_box(ax2, arnold_cat_map, "Suspension of Arnold's Cat Map\n(Torus Bundle / Solvmanifold)")

plt.tight_layout()
plt.savefig("suspension_2d_flows.pdf")


import numpy as np
import matplotlib.pyplot as plt

def bakers_map_flow(points):
    x, y = points[:, 0], points[:, 1]
    mask = x < 0.5
    
    # Baker's Map 변환
    x_next = np.where(mask, 2 * x, 2 * x - 1)
    y_next = np.where(mask, 0.5 * y, 0.5 * y + 0.5)
    
    return np.stack([x_next, y_next], axis=1)

def plot_multistep_suspension(n_steps=3):
    # 1. 초기 데이터 (스마일리 대신 단순한 사각형 두 개로 구분)
    # 왼쪽(Blue), 오른쪽(Red) 영역
    n_p = 2000
    
    # 왼쪽 덩어리
    x_L = np.random.uniform(0.1, 0.4, n_p)
    y_L = np.random.uniform(0.1, 0.9, n_p)
    points_L = np.stack([x_L, y_L], axis=1)
    
    # 오른쪽 덩어리
    x_R = np.random.uniform(0.6, 0.9, n_p)
    y_R = np.random.uniform(0.1, 0.9, n_p)
    points_R = np.stack([x_R, y_R], axis=1)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 현재 상태 저장
    curr_L = points_L.copy()
    curr_R = points_R.copy()
    
    for step in range(n_steps):
        # 다음 상태 계산
        next_L = bakers_map_flow(curr_L)
        next_R = bakers_map_flow(curr_R)
        
        # 시각화 (선형 보간으로 흐름 표현)
        # step ~ step+1 사이를 채움
        
        # 보기 좋게 점들을 일부만 샘플링해서 선을 그림 (너무 많으면 복잡함)
        samples = 100 
        for i in range(0, n_p, n_p//samples):
            # Blue Flow
            z_start = step
            z_end = step + 1
            ax.plot([curr_L[i,0], next_L[i,0]], 
                    [curr_L[i,1], next_L[i,1]], 
                    [z_start, z_end], color='blue', alpha=0.3, linewidth=0.5)
            
            # Red Flow
            ax.plot([curr_R[i,0], next_R[i,0]], 
                    [curr_R[i,1], next_R[i,1]], 
                    [z_start, z_end], color='red', alpha=0.3, linewidth=0.5)
            
        # 각 단계의 단면(Section)을 점으로 표시
        # t = step (바닥)
        ax.scatter(curr_L[:,0], curr_L[:,1], step, c='blue', s=1, alpha=0.5)
        ax.scatter(curr_R[:,0], curr_R[:,1], step, c='red', s=1, alpha=0.5)
        
        # 상태 업데이트
        curr_L = next_L
        curr_R = next_R

    # 마지막 천장 표시
    ax.scatter(curr_L[:,0], curr_L[:,1], n_steps, c='blue', s=1, alpha=0.8)
    ax.scatter(curr_R[:,0], curr_R[:,1], n_steps, c='red', s=1, alpha=0.8)
    
    ax.set_title(f"Suspension Flow of Baker's Map ({n_steps} Iterations)")
    ax.set_xlabel('X (Stretch)')
    ax.set_ylabel('Y (Compress)')
    ax.set_zlabel('Time (t)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, n_steps)
    
    # 시각적 가이드: 각 정수 시간(t=1, 2...)에 투명한 판 그리기
    for i in range(1, n_steps):
        xx, yy = np.meshgrid([0,1], [0,1])
        ax.plot_surface(xx, yy, np.full_like(xx, i), alpha=0.1, color='gray')

    plt.tight_layout()
    plt.savefig("suspension_bakers_map_multistep.pdf") 

plot_multistep_suspension(n_steps=3)