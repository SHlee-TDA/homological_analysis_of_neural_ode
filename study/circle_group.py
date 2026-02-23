import numpy as np
import matplotlib.pyplot as plt

class CircleGroup:
    """
    This class is a implementation of the circle group 
    S^1 = [0,1]/~ where ~ denotes the identification of 0 and 1
    """
    def __init__(self, value: float):
        self.value = value % 1 # Normalize the input value as 0 <= value < 1
    
    def __repr__(self):
        return f"{self.value} in S^1"
    
    def __add__(self, other):
        if isinstance(other, CircleGroup):
            return CircleGroup(self.value + other.value)
        elif isinstance(other, (int, float)):
            return CircleGroup(self.value + other)
        else:
            raise TypeError("Unsupported operand type")
        
    @staticmethod
    def distance(x, y):
        # Metric d(x,y) = min(|x-y|, 1 - |x-y|)
        val_x = x.value if isinstance(x, CircleGroup) else x % 1
        val_y = y.value if isinstance(y, CircleGroup) else y % 1
        diff = np.abs(val_x - val_y)
        return np.minimum(diff, 1.0 - diff)

def generate_orbit(alpha, n_steps, x0=0.0):
    orbit = []
    current_point = CircleGroup(x0)
    
    for _ in range(n_steps):
        orbit.append(current_point.value)
        current_point = current_point + alpha
        
    return np.array(orbit)

    
def main():
    a = CircleGroup(0.3)
    b = CircleGroup(0.8)
    c = CircleGroup(1.2)
    
    print("a:", a)
    print("b:", b)
    print("c:", c)
    
    print("a + b:", a + b)
    print("a + 0.5:", a + 0.5)
    
    print("Distance between a and b:", CircleGroup.distance(a, b))
    print("Distance between a and c:", CircleGroup.distance(a, c))

        
    
if __name__ == '__main__':
    main()
    
    # 파라미터 설정
    n_steps = 1000
    alpha_rational = 1/5.0
    alpha_irrational = np.sqrt(2) # 실제로는 부동소수점 근사값

    orbit_rational = generate_orbit(alpha_rational, n_steps)
    orbit_irrational = generate_orbit(alpha_irrational, n_steps)

    # 시각화 (Phase Space Plot)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': 'polar'})

    # 유리수 회전 시각화
    theta_r = 2 * np.pi * orbit_rational
    ax[0].plot(theta_r, np.ones_like(theta_r), 'o', markersize=5, alpha=0.5)
    ax[0].set_title(f"Rational Rotation (alpha=1/5)\nPeriodic Orbit")
    ax[0].set_yticks([])

    # 무리수 회전 시각화
    theta_ir = 2 * np.pi * orbit_irrational
    ax[1].plot(theta_ir, np.ones_like(theta_ir), 'o', markersize=2, alpha=0.3)
    ax[1].set_title(f"Irrational Rotation (alpha=sqrt(2))\nDense Orbit")
    ax[1].set_yticks([])

    plt.savefig("circle_group_orbits.pdf")
    