import torch
import torch.nn as nn

class HNN(nn.Module):
    def __init__(self, input_dim, 
                 hidden_dim, hidden_layers, activation):
        super(HNN, self).__init__()
        # input (q, p) consists of positions and momentums.
        # so, the dimension must be even.
        assert input_dim % 2 == 0, "Input dimension must be even."
        
        self.input_dim = input_dim
        self.output_dim = 1 # Hamiltonian is scalar
        
        self.activation = self._get_activation(activation)

        # Stack hidden layers
        layers = []
        for i in range(hidden_layers):
            layers.append(nn.Linear(self.input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(self.activation)
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x (batch, input_dim) -> H (batch, 1)
        H = self.network(x)
        return H
            
    def symplectic_gradient(self, x):
        x = x.clone().detach().requires_grad_(True)
        H = self.forward(x)
        
        dHdx = torch.autograd.grad(
            outputs=H,
            inputs=x,
            grad_outputs=torch.ones_like(H),
            create_graph=True            
        )[0]
        
        # Split gradients into q and p components
        half_dim = self.input_dim // 2
        dHdq = dHdx[:, :half_dim]
        dHdp = dHdx[:, half_dim:]
        
        symplectic_grad = torch.cat([dHdp, -dHdq], dim=1)
        return symplectic_grad
    
    def compute_loss(self, x, dxdt):
        # symp_grad[0] = dHdp, sym_grad[1] = -dH/dq
        symp_grad = self.symplectic_gradient(x)
        loss = torch.mean((symp_grad - dxdt)**2)
        return loss
    
    def _get_activation(self, name):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {name}")
        
        
class BaselineNN(nn.Module):
    """
    Directly estimates time derivatives (dq/dt, dp/dt) from state (q, p).
    Does NOT enforce Hamiltonian mechanics or energy conservation.
    """
    def __init__(self, input_dim, hidden_dim, hidden_layers, activation):
        super(BaselineNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = input_dim # Output is vector field (same dim as input)
        
        self.activation = self._get_activation(activation)

        layers = []
        for i in range(hidden_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(self.activation)
        
        # 마지막 레이어: 직접 미분값 예측
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    # Trainer와 호환성을 위해 HNN_loss와 같은 인터페이스를 맞춤
    def compute_loss(self, x, dxdt):
        # x: input state (q, p)
        # dxdt: target vector field (dq/dt, dp/dt)
        
        dxdt_pred = self.forward(x)
        
        # Simple MSE Loss
        loss = torch.mean((dxdt_pred - dxdt) ** 2)
        return loss
    
    def _get_activation(self, name):
        # HNN과 동일한 활성화 함수 로직 사용
        if name == 'tanh': return nn.Tanh()
        elif name == 'relu': return nn.ReLU()
        elif name == 'sigmoid': return nn.Sigmoid()
        else: raise ValueError(f"Unsupported activation: {name}")