# hodge_analyzer.py

import argparse
from itertools import combinations
import time

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from scipy.sparse.linalg import eigsh  
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# 1. JAX로 최적화된 핵심 계산 함수들
# --------------------------------------------------------------------------

@jax.jit
def compute_distance_matrix_jax(point_cloud: jnp.ndarray) -> jnp.ndarray:
    diff = point_cloud[:, jnp.newaxis, :] - point_cloud[jnp.newaxis, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1))

@jax.jit
def compute_weights_jax(point_cloud: jnp.ndarray, epsilon: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """간선과 면의 가중치 텐서를 한 번에 계산합니다."""
    # 간선 가중치
    dist_matrix = compute_distance_matrix_jax(point_cloud)
    edge_weights = jnp.exp(-dist_matrix**2 / (4 * epsilon))
    
    # 면 가중치
    T1 = jnp.einsum('ij,ik->ijk', edge_weights, edge_weights)
    T2 = jnp.einsum('ij,jk->ijk', edge_weights, edge_weights)
    T3 = jnp.einsum('ik,jk->ijk', edge_weights, edge_weights)
    face_weights = (T1 + T2 + T3) / 3.0
    
    return edge_weights, face_weights

# JIT 외부: 정적 위상 구조 계산 (한 번만 수행)
def build_boundary_operators_static(n_points: int) -> tuple:
    edges = np.array(sorted(list(combinations(range(n_points), 2))))
    triangles = np.array(sorted(list(combinations(range(n_points), 3))))
    edge_map = {tuple(edge): i for i, edge in enumerate(edges)}
    num_edges, num_triangles = len(edges), len(triangles)
    
    # d1
    d1_col = np.arange(num_edges).repeat(2)
    d1_row = edges.flatten()
    d1_data = np.tile([-1, 1], num_edges)
    d1_indices = jnp.array(np.vstack([d1_row, d1_col]).T)
    d1 = BCOO((jnp.array(d1_data), d1_indices), shape=(n_points, num_edges))

    # d2
    d2_col = np.arange(num_triangles).repeat(3)
    d2_row = np.array([idx for i,j,k in triangles for idx in (edge_map[(i,j)], edge_map[(j,k)], edge_map[(i,k)])])
    d2_data = np.tile([1, 1, -1], num_triangles)
    d2_indices = jnp.array(np.vstack([d2_row, d2_col]).T)
    d2 = BCOO((jnp.array(d2_data), d2_indices), shape=(num_edges, num_triangles))
                    
    return d1, d2, jnp.array(edges), jnp.array(triangles)

@jax.jit
def assemble_hodge_laplacian_jax(d1: BCOO, d2: BCOO, edges: jnp.ndarray, triangles: jnp.ndarray,
                                 edge_weights_tensor: jnp.ndarray, face_weights_tensor: jnp.ndarray) -> BCOO:
    w1_vals = edge_weights_tensor[edges[:, 0], edges[:, 1]]
    w2_vals = face_weights_tensor[triangles[:, 0], triangles[:, 1], triangles[:, 2]]
    
    W1 = BCOO.fromdense(jnp.diag(w1_vals))
    W1_inv = BCOO.fromdense(jnp.diag(jnp.where(w1_vals > 1e-12, 1.0/w1_vals, 0)))
    W2 = BCOO.fromdense(jnp.diag(w2_vals))
    
    L_down = d1.transpose() @ d1 @ W1
    L_up = W1_inv @ d2 @ W2 @ d2.transpose()
    
    return L_up + L_down

# --------------------------------------------------------------------------
# 2. 메인 파이프라인 함수
# --------------------------------------------------------------------------

def estimate_betti_number_from_point_cloud(point_cloud: np.ndarray, epsilon: float, k: int, threshold: float = 1e-8) -> tuple:
    """
    포인트 클라우드로부터 Hodge Laplacian을 계산하고 Betti 수를 추정하는 전체 파이프라인.
    """
    n_points = point_cloud.shape[0]
    print(f"총 {n_points}개의 포인트로 계산을 시작합니다...")
    
    # 1. 정적 위상 구조 계산 (Python/NumPy)
    start_time = time.time()
    d1, d2, edges, triangles = build_boundary_operators_static(n_points)
    print(f"경계 연산자 생성 완료. ({time.time() - start_time:.2f}초)")
    
    # 2. 가중치 계산 (JAX JIT 컴파일)
    start_time = time.time()
    point_cloud_jax = jnp.array(point_cloud)
    edge_weights, face_weights = compute_weights_jax(point_cloud_jax, epsilon)
    edge_weights.block_until_ready() # JAX의 비동기 실행 완료 대기
    print(f"가중치 텐서 계산 완료. ({time.time() - start_time:.2f}초)")
    
    # 3. 라플라시안 조립 (JAX JIT 컴파일)
    start_time = time.time()
    Delta_1 = assemble_hodge_laplacian_jax(d1, d2, edges, triangles, edge_weights, face_weights)
    Delta_1.block_until_ready()
    print(f"Hodge Laplacian 조립 완료. ({time.time() - start_time:.2f}초)")
    
    # 4. 고유값 계산 및 Betti 수 추정
    start_time = time.time()
    Delta_1_dense = Delta_1.todense()
    eigenvalues = eigsh(Delta_1_dense, k=k, which='SM', return_eigenvectors=False)
    eigenvalues = np.sort(eigenvalues)
    estimated_betti = np.sum(eigenvalues < threshold)
    print(f"고유값 계산 완료. ({time.time() - start_time:.2f}초)")
    
    return estimated_betti, eigenvalues

# --------------------------------------------------------------------------
# 3. 테스트용 데이터 생성 및 시각화 함수
# --------------------------------------------------------------------------

def generate_torus_test_case(n_samples_per_axis: int) -> np.ndarray:
    """테스트용 토러스 포인트 클라우드를 생성합니다."""
    ticks = np.linspace(0, 2 * np.pi, n_samples_per_axis)
    u, v = np.meshgrid(ticks, ticks)
    u, v = u.flatten(), v.flatten()
    return np.vstack([np.cos(u), np.sin(u), np.cos(v), np.sin(v)]).T

def plot_spectrum(eigenvalues, estimated_betti):
    """결과 고유값 스펙트럼을 시각화합니다."""
    plt.figure(figsize=(10, 6))
    plt.plot(eigenvalues, 'o-')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Eigenvalue Spectrum of $\Delta_1$ (Estimated $\\beta_1$ = {estimated_betti})', fontsize=16)
    plt.xlabel('Eigenvalue Index (sorted)', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.grid(True)
    plt.show()

# --------------------------------------------------------------------------
# 4. 커맨드 라인 실행을 위한 메인 블록
# --------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate the first Betti number of a point cloud using Hodge Laplacian.")
    parser.add_argument('--infile', type=str, help="Path to the .npy file containing the point cloud.")
    parser.add_argument('--epsilon', type=float, required=True, help="Epsilon (bandwidth) for the Gaussian kernel.")
    parser.add_argument('--k', type=int, default=30, help="Number of smallest eigenvalues to compute.")
    parser.add_argument('--plot', action='store_true', help="If set, display the eigenvalue spectrum plot.")
    
    args = parser.parse_args()

    if args.infile:
        print(f"파일에서 포인트 클라우드를 불러옵니다: {args.infile}")
        point_cloud = np.load(args.infile)
    else:
        print("입력 파일이 없습니다. 토러스 테스트 케이스를 실행합니다...")
        n_axis = 25 # 25*25 = 625 points
        point_cloud = generate_torus_test_case(n_axis)
        print(f"테스트용 토러스 생성 완료 ({n_axis}x{n_axis}={n_axis**2} points).")

    # 메인 파이프라인 실행
    betti, eigenvalues = estimate_betti_number_from_point_cloud(point_cloud, args.epsilon, args.k)

    print("\n--- 최종 결과 ---")
    print(f"추정된 첫 번째 Betti 수 (β₁): {betti}")

    if args.plot:
        plot_spectrum(eigenvalues, betti)