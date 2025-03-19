import torch
import torch.nn as nn
import math

class LTanh(nn.Module):
    """
    Layer-wise locally Adaptive Tanh activation function
    """
    def __init__(self, n=10):
        super(LTanh, self).__init__()
        self.n = n
        self.a = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        return torch.tanh(self.n * self.a * x)

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for better gradient flow
    """
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation = LTanh()
        
        # Skip connection
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        # Initialize weights using Glorot normal initialization
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        
        if isinstance(self.shortcut, nn.Linear):
            nn.init.xavier_normal_(self.shortcut.weight)
            nn.init.zeros_(self.shortcut.bias)
            
    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + self.shortcut(residual)
        out = self.activation(out)
        return out

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for heat transfer modeling in LMD process
    """
    def __init__(self, num_neurons=30, num_blocks=1, input_dim=4, output_dim=1):
        super(PINN, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, num_neurons)
        self.input_activation = LTanh()
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.hidden_layers.append(ResidualBlock(num_neurons, num_neurons))
        
        # Output layer
        self.output_layer = nn.Linear(num_neurons, output_dim)
        
        # Initialize weights
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"PINN model has {self.num_params} trainable parameters")
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, 4] (x, y, z, t coordinates)
            
        Returns:
            Network output tensor of shape [batch_size, 1]
        """
        # Input layer
        x = self.input_activation(self.input_layer(x))
        
        # Hidden layers with residual blocks
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Output layer - using sigmoid to constrain output between 0 and 1
        # This helps with scaling the temperature appropriately
        x = torch.sigmoid(self.output_layer(x))
        
        return x
