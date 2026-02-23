import numpy as np
import matplotlib.pyplot as plt

def bakers_map(points, a=0.25):
    x_curr, y_curr = points[:, 0], points[:, 1]
    mask = x_curr < 0.5
    x_next = np.where(mask, 2 * x_curr, 2 * x_curr - 1)
    y_next = np.where(mask, a * y_curr, a * y_curr + 0.5)
    return np.stack([x_next, y_next], axis=1)


# 1. 초기 데이터 생성 (스마일리 얼굴 모양)
def create_smiley(n_points=10000):
    # 얼굴 윤곽선 (원)
    theta = np.linspace(0, 2*np.pi, n_points//2)
    x = 0.5 + 0.4 * np.cos(theta)
    y = 0.5 + 0.4 * np.sin(theta)
    
    # 눈
    theta_eye = np.linspace(0, 2*np.pi, 100)
    # 왼쪽 눈
    x = np.append(x, 0.35 + 0.05 * np.cos(theta_eye))
    y = np.append(y, 0.65 + 0.05 * np.sin(theta_eye))
    # 오른쪽 눈
    x = np.append(x, 0.65 + 0.05 * np.cos(theta_eye))
    y = np.append(y, 0.65 + 0.05 * np.sin(theta_eye))
    
    # 입 (반원)
    theta_smile = np.linspace(np.pi+0.5, 2*np.pi-0.5, n_points//4)
    x = np.append(x, 0.5 + 0.2 * np.cos(theta_smile))
    y = np.append(y, 0.5 + 0.2 * np.sin(theta_smile))
    
    return np.stack([x, y], axis=1)


points = create_smiley()
initial_colors = np.where(points[:, 0] < 0.5, 'blue', 'red')

# 2. 시각화 (0, 1, 2, 3회 반복 결과)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
iterations = [0, 1, 2, 7]
points_curr = points.copy()
current_iter = 0

for i, ax in enumerate(axes):
    target_iter = iterations[i]
    
    # 필요한 만큼 반복 진행
    while current_iter < target_iter:
        points_curr = bakers_map(points_curr, a=0.49)
        current_iter += 1
    
    ax.scatter(points_curr[:, 0], points_curr[:, 1], c=initial_colors, s=0.5, alpha=0.6)
    ax.set_title(f'Iteration {target_iter}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("bakers_map_iterations.pdf")

n_points = 100000  # 점의 개수 (많은 점을 사용)
n_iterations = 12 # 반복 횟수 (어트랙터에 수렴하도록 충분히)
a_val = 0.4       # a < 0.5 (소산계, Cantor set 구조 생성)

# --- 초기 점 생성 및 반복 ---
# 0~1 사이의 랜덤한 점 생성
points = np.random.rand(n_points, 2)

# Baker's Map 반복 적용
for _ in range(n_iterations):
    points = bakers_map(points, a=a_val)

# --- 시각화 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 1. 전체 어트랙터 모습
ax1.scatter(points[:, 0], points[:, 1], s=0.1, c='black', alpha=0.5)
ax1.set_title(f'Baker\'s Map Strange Attractor (a={a_val}, N={n_points}, Iter={n_iterations})')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_aspect('equal')

# 2. 확대 (Zoom-in) - 칸토어 집합 구조 확인
# y축 하단부를 확대하여 자기 유사성 확인
zoom_y_range = [0, 0.1]
ax2.scatter(points[:, 0], points[:, 1], s=0.5, c='black', alpha=0.5) # 확대 시 점 크기 약간 키움
ax2.set_title('Zoom-in (y-axis) showing Cantor Set Structure')
ax2.set_xlim(0, 1)
ax2.set_ylim(zoom_y_range)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
# 확대 영역의 aspect ratio를 맞춰줌
ax2.set_aspect(1 / (zoom_y_range[1] - zoom_y_range[0])) 

plt.tight_layout()
plt.savefig("bakers_map_strange_attractor.pdf")