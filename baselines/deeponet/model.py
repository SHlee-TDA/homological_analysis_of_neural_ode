import torch
import torch.nn as nn
import torch
import torch.nn as nn

class FNN(nn.Module):
    """Fully Connected Neural Network (MLP)"""
    def __init__(self, layer_sizes, activation=nn.Tanh()):
        super(FNN, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2: 
                layers.append(activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SimpleDeepONet(nn.Module):
    def __init__(self, 
                 branch_input_dim=2,  # (q0, p0)
                 trunk_input_dim=1,   # t
                 hidden_depth=3,      # 은닉층 개수
                 hidden_width=64,     # 은닉층 너비
                 latent_dim=64,       # 기저 함수 개수 (p)
                 output_dim=2):       # (q_t, p_t)
        super(SimpleDeepONet, self).__init__()
        
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        
        branch_layers = [branch_input_dim] + [hidden_width]*hidden_depth + [latent_dim * output_dim]
        self.branch_net = FNN(branch_layers, activation=nn.Tanh())
        
        
        trunk_layers = [trunk_input_dim] + [hidden_width]*hidden_depth + [latent_dim]
        self.trunk_net = FNN(trunk_layers, activation=nn.Tanh())
        
        # Bias parameter (각 출력 차원마다 하나의 bias)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, u, y):
        """
        u: Branch input (Batch_Size, branch_input_dim) ->  (N, 2)
        y: Trunk input  (Batch_Size, trunk_input_dim)  ->  (N, 1)
        """
        # 1. Branch Net Forward
        # B_out shape: (Batch, latent_dim * output_dim)
        B_out = self.branch_net(u)
        
        # 2. Trunk Net Forward
        # T_out shape: (Batch, latent_dim)
        T_out = self.trunk_net(y)
        
        # 3. Reshape for Merge
        # Branch 출력을 (Batch, Output, Latent)로 변환
        B_out = B_out.view(-1, self.output_dim, self.latent_dim)
        
        # Trunk 출력을 (Batch, 1, Latent)로 변환 (Broadcasting 준비)
        T_out = T_out.unsqueeze(1)
        
        # 4. Merge (Dot Product) 
        # Element-wise 곱을 수행한 후 Latent 차원(dim=-1)에 대해 합산
        # (Batch, Out, Latent) * (Batch, 1, Latent) -> (Batch, Out, Latent) -> Sum -> (Batch, Out)
        outputs = torch.sum(B_out * T_out, dim=-1)
        
        # Bias 추가
        outputs = outputs + self.bias
        
        return outputs