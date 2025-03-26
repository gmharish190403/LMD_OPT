import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
from torch.autograd import grad
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

# PINN Model matching exactly the paper's architecture (30 neurons, 1 residual block)
class PINN(nn.Module):
    """
    Physics-Informed Neural Network for LMD process temperature prediction
    - Exact implementation from the paper (30 neurons, 1 residual block)
    """
    def __init__(self, input_dim=4, hidden_dim=30, output_dim=1):
        super(PINN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input normalization 
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layer 1
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Hidden layer 2 (residual block)
        self.hidden2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2_2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights using Xavier initialization
        self._init_weights()
        
    def _init_weights(self):
        # Initialize input layer
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        
        # Initialize hidden layers
        nn.init.xavier_normal_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        
        # Initialize residual block
        nn.init.xavier_normal_(self.hidden2_1.weight)
        nn.init.zeros_(self.hidden2_1.bias)
        nn.init.xavier_normal_(self.hidden2_2.weight)
        nn.init.zeros_(self.hidden2_2.bias)
        
        # Initialize output layer
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)
        
    def set_input_stats(self, mean, std):
        """Set the input normalization statistics"""
        self.input_mean.data.copy_(torch.tensor(mean, dtype=torch.float32))
        self.input_std.data.copy_(torch.tensor(std, dtype=torch.float32))
    
    def forward(self, x):
        # Normalize input (x, y, z, t)
        x_norm = (x - self.input_mean) / self.input_std
        
        # Input layer with tanh activation
        x = torch.tanh(self.input_layer(x_norm))
        
        # Hidden layer 1
        x = torch.tanh(self.hidden1(x))
        
        # Hidden layer 2 (residual block)
        residual = x
        x = torch.tanh(self.hidden2_1(x))
        x = self.hidden2_2(x)
        x = x + residual  # Residual connection
        x = torch.tanh(x)
        
        # Output layer with sigmoid to bound output between 0 and 1
        x = torch.sigmoid(self.output_layer(x))
        
        return x

# Physics implementation
class Physics:
    def __init__(self, params, scale_u=3000.0):
        """
        Initialize the physics module with process parameters
        Exactly matching the parameters in Table 2 of the paper
        """
        self.params = params
        self.scale_u = scale_u  # Temperature scaling factor (T = T0 + u*scale_u)
        
        # Convert domain parameters to tensor constants for faster computation
        self.Lx = torch.tensor(params['Lx'])
        self.Ly = torch.tensor(params['Ly'])
        self.Lz = torch.tensor(params['Lz'])
        self.T0 = torch.tensor(params['T0'])
        self.t1 = torch.tensor(params['t1'])
        self.t2 = torch.tensor(params['t2'])
        
        # Laser normalization factor exactly as in the paper
        self.laser_norm_factor = (8 * np.sqrt(3) * params['eta'] * params['P']) / (np.pi * np.sqrt(np.pi) * params['Ra'] * params['Rb'] * params['Rc'])
        
        # Stefan-Boltzmann constant
        self.sigma_sb = 5.67e-8
        
        # Reference value for PDE normalization from the paper
        self.Q_max = 5.0e11
        
        # For monitoring during training
        self.print_freq = 500
    
    def unscale_temperature(self, u_scaled):
        """Convert normalized temperature [0,1] to physical temperature [K]"""
        return self.T0 + u_scaled * self.scale_u
    
    def thermal_conductivity(self, T):
        """
        Temperature-dependent thermal conductivity exactly as in Table 2 of the paper
        
        Args:
            T: Temperature in Kelvin
            
        Returns:
            k: Thermal conductivity in W/(m·K)
        """
        # Thermal conductivity as a function of temperature (from paper Table 2)
        # k(T) = 7.17 + 2.37e-2*T - 6.87e-6*T^2 (from 293.15K to 1773.15K)
        # k(T) = 31.9 (from 1773.15K to 3000K)
        
        threshold = 1773.15
        k_low = 7.17 + 2.37e-2 * T - 6.87e-6 * T**2
        k_high = torch.full_like(T, 31.9)
        
        # Use torch.where for stability
        k = torch.where(T < threshold, k_low, k_high)
        
        # Ensure physical bounds
        k = torch.clamp(k, min=7.0, max=40.0)
        return k
    
    def specific_heat(self, T):
        """
        Temperature-dependent specific heat capacity exactly as in Table 2 of the paper
        
        Args:
            T: Temperature in Kelvin
            
        Returns:
            Cp: Specific heat capacity in J/(kg·K)
        """
        # Specific heat as a function of temperature (from paper Table 2)
        # Cp(T) = 435.5 + 0.215*T - 4.3e-5*T^2 (from 293.15K to 1773.15K)
        # Cp(T) = 700 (from 1773.15K to 3000K)
        
        threshold = 1773.15
        cp_low = 435.5 + 0.215 * T - 4.3e-5 * T**2
        cp_high = torch.full_like(T, 700.0)
        
        # Use torch.where for stability
        cp = torch.where(T < threshold, cp_low, cp_high)
        
        # Ensure physical bounds
        cp = torch.clamp(cp, min=400.0, max=800.0)
        return cp
    
    def laser_heat_source(self, x, y, z, t):
        """
        Gaussian laser heat source following the paper's equation
        """
        # Get parameters
        v = self.params['v']
        Ra = self.params['Ra']
        Rb = self.params['Rb']
        Rc = self.params['Rc']
        P = self.params['P']
        eta = self.params['eta']
        
        # Starting position
        x0 = 0.0
        y0 = self.params['Ly'] / 2  # Center of domain in y
        z0 = 0.0  # Top surface
        
        # Laser is only active during deposition time
        is_active = t <= self.t1
        
        # Normalized squared distance
        r_squared = ((x - (v*t + x0))/Ra)**2 + ((y - y0)/Rb)**2 + ((z - z0)/Rc)**2
        
        # Coefficient exactly as in the paper
        coef = (6 * np.sqrt(3) * eta * P) / (np.pi * np.sqrt(np.pi) * Ra * Rb * Rc)
        
        # Gaussian heat source with exponent of 3 as per paper
        Q = coef * torch.exp(-3 * r_squared)
        
        # Zero out when laser is inactive
        Q = torch.where(is_active, Q, torch.zeros_like(Q))
        
        return Q
    
    def compute_derivatives(self, model, x, y, z, t):
        """
        Compute spatial and temporal derivatives using automatic differentiation
        
        Args:
            model: PINN model
            x, y, z, t: Spatial and temporal coordinates
            
        Returns:
            Temperature and its derivatives needed for physics equations
        """
        # Create tensors that require gradient
        x_tensor = x.clone().requires_grad_(True)
        y_tensor = y.clone().requires_grad_(True)
        z_tensor = z.clone().requires_grad_(True)
        t_tensor = t.clone().requires_grad_(True)
        
        # Stack inputs
        inputs = torch.stack([x_tensor, y_tensor, z_tensor, t_tensor], dim=1)
        
        # Forward pass
        u = model(inputs)
        
        # Physical temperature
        T = self.unscale_temperature(u)
        
        # First derivatives
        dT_dx = grad(outputs=T, inputs=x_tensor, 
                    grad_outputs=torch.ones_like(T),
                    create_graph=True)[0]
        
        dT_dy = grad(outputs=T, inputs=y_tensor, 
                    grad_outputs=torch.ones_like(T),
                    create_graph=True)[0]
        
        dT_dz = grad(outputs=T, inputs=z_tensor, 
                    grad_outputs=torch.ones_like(T),
                    create_graph=True)[0]
        
        dT_dt = grad(outputs=T, inputs=t_tensor, 
                    grad_outputs=torch.ones_like(T),
                    create_graph=True)[0]
        
        # Second derivatives
        d2T_dx2 = grad(outputs=dT_dx, inputs=x_tensor, 
                      grad_outputs=torch.ones_like(dT_dx),
                      create_graph=True)[0]
        
        d2T_dy2 = grad(outputs=dT_dy, inputs=y_tensor, 
                      grad_outputs=torch.ones_like(dT_dy),
                      create_graph=True)[0]
        
        d2T_dz2 = grad(outputs=dT_dz, inputs=z_tensor, 
                      grad_outputs=torch.ones_like(dT_dz),
                      create_graph=True)[0]
        
        return u, T, dT_dt, dT_dx, dT_dy, dT_dz, d2T_dx2, d2T_dy2, d2T_dz2
    
    def pde_residual(self, model, x, y, z, t, epoch=None):
        """
        Calculate the residual of the heat equation
        Matching Equation 5 in the paper
        
        Args:
            model: PINN model
            x, y, z, t: Spatial and temporal coordinates
            epoch: Current training epoch (for debugging)
            
        Returns:
            residual: PDE residual
        """
        # Compute all derivatives
        u, T, dT_dt, dT_dx, dT_dy, dT_dz, d2T_dx2, d2T_dy2, d2T_dz2 = self.compute_derivatives(model, x, y, z, t)
        
        # Get temperature-dependent material properties
        k = self.thermal_conductivity(T)
        Cp = self.specific_heat(T)
        rho = self.params['rho']
        
        # Compute heat source term
        Q = self.laser_heat_source(x, y, z, t)
        
        # Laplacian term (∇²T)
        laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2
        
        # Heat equation residual (normalized by Q_max as in the paper)
        residual = (rho * Cp * dT_dt - k * laplacian - Q) / self.Q_max
        
        # Print debugging info occasionally
        if epoch is not None and (epoch + 1) % self.print_freq == 0:
            with torch.no_grad():
                print(f"Epoch {epoch+1} | T min: {T.min().item():.2f}K, max: {T.max().item():.2f}K")
                print(f"Epoch {epoch+1} | Laplacian min: {laplacian.min().item():.2e}, max: {laplacian.max().item():.2e}")
                print(f"Epoch {epoch+1} | Heat source min: {Q.min().item():.2e}, max: {Q.max().item():.2e}")
                print(f"Epoch {epoch+1} | Residual min: {residual.min().item():.2e}, max: {residual.max().item():.2e}")
        
        return residual
    
    def ic_residual(self, model, x, y, z, t):
        """
        Initial condition residual (Equation 9 in the paper)
        
        Args:
            model: PINN model
            x, y, z, t: Spatial and temporal coordinates
            
        Returns:
            residual: IC residual
        """
        # Forward pass
        inputs = torch.stack([x, y, z, t], dim=1)
        u = model(inputs)
        
        # Convert to physical temperature
        T = self.unscale_temperature(u)
        
        # Initial condition residual (normalized)
        return (T - self.T0) / self.scale_u
    
    def bc_residual(self, model, x, y, z, t):
        """
        Boundary condition residual (Equations 3a-3c in the paper)
        
        Args:
            model: PINN model
            x, y, z, t: Spatial and temporal coordinates
            
        Returns:
            residual: BC residual
        """
        # Compute derivatives
        u, T, dT_dt, dT_dx, dT_dy, dT_dz, d2T_dx2, d2T_dy2, d2T_dz2 = self.compute_derivatives(model, x, y, z, t)
        
        # Material properties
        k = self.thermal_conductivity(T)
        h = self.params['h']  # Convection coefficient
        epsilon = self.params['epsilon']  # Emissivity
        
        # Boundary detection
        tol = 1e-5
        is_x_min = torch.abs(x) < tol
        is_x_max = torch.abs(x - self.Lx) < tol
        is_y_min = torch.abs(y) < tol
        is_y_max = torch.abs(y - self.Ly) < tol
        is_z_min = torch.abs(z) < tol
        is_z_max = torch.abs(z - self.Lz) < tol
        
        # Initialize boundary residual
        bc_residual = torch.zeros_like(T)
        
        # Apply boundary conditions exactly as in the paper
        
        # x=0: Insulated (dT/dx = 0)
        bc_residual = torch.where(is_x_min, dT_dx, bc_residual)
        
        # x=Lx: Convection+radiation
        conv_rad_x = -k * dT_dx - h * (T - self.T0) - epsilon * self.sigma_sb * (T**4 - self.T0**4)
        bc_residual = torch.where(is_x_max, conv_rad_x, bc_residual)
        
        # y=0 and y=Ly: Insulated (dT/dy = 0)
        bc_residual = torch.where(is_y_min | is_y_max, dT_dy, bc_residual)
        
        # z=0: Insulated (dT/dz = 0)
        bc_residual = torch.where(is_z_min, dT_dz, bc_residual)
        
        # z=Lz: Convection+radiation
        conv_rad_z = -k * dT_dz - h * (T - self.T0) - epsilon * self.sigma_sb * (T**4 - self.T0**4)
        bc_residual = torch.where(is_z_max, conv_rad_z, bc_residual)
        
        # Normalize residual as done in the paper
        return bc_residual / self.Q_max

def generate_training_points(params, n_pde, n_ic, n_bc, device):
    """
    Generate training points with sampling strategy from the paper (Section 2.5)
    
    The paper uses a total of approximately 30,000 points
    """
    # Domain dimensions
    Lx, Ly, Lz = params['Lx'], params['Ly'], params['Lz']
    t1, t2 = params['t1'], params['t2']
    t_max = t1 + t2
    v = params['v']
    
    # 1. PDE points - focused sampling around laser path and high gradient regions
    # Uniform points in domain (30%)
    n_uniform = int(0.3 * n_pde)
    x_uniform = torch.rand(n_uniform, 1, device=device) * Lx
    y_uniform = torch.rand(n_uniform, 1, device=device) * Ly
    z_uniform = torch.rand(n_uniform, 1, device=device) * Lz
    t_uniform = torch.rand(n_uniform, 1, device=device) * t_max
    
    # Focused points around laser path (50%) - where high gradients are expected
    n_laser = int(0.5 * n_pde)
    
    # Time points biased toward deposition phase
    # Using power law distribution to concentrate more points during deposition
    t_laser = torch.rand(n_laser, 1, device=device).pow(0.5) * t1
    
    # Laser position (along x-axis) with small noise
    x_laser = v * t_laser + 0.001 * torch.randn(n_laser, 1, device=device)
    x_laser = torch.clamp(x_laser, 0, Lx)
    
    # Concentrated around laser center in y-direction
    y_laser = Ly/2 + 0.001 * torch.randn(n_laser, 1, device=device)
    y_laser = torch.clamp(y_laser, 0, Ly)
    
    # More points near surface in z-direction
    z_laser = 0.0003 * torch.rand(n_laser, 1, device=device).pow(3)
    z_laser = torch.clamp(z_laser, 0, Lz)
    
    # Surface points (20%) for better surface temperature prediction
    n_surface = n_pde - n_uniform - n_laser
    x_surface = torch.rand(n_surface, 1, device=device) * Lx
    y_surface = torch.rand(n_surface, 1, device=device) * Ly
    z_surface = torch.zeros(n_surface, 1, device=device)  # z=0 surface
    t_surface = torch.rand(n_surface, 1, device=device) * t_max
    
    # Combine all PDE points
    x_pde = torch.cat([x_uniform, x_laser, x_surface])
    y_pde = torch.cat([y_uniform, y_laser, y_surface])
    z_pde = torch.cat([z_uniform, z_laser, z_surface])
    t_pde = torch.cat([t_uniform, t_laser, t_surface])
    
    # Stack as input tensor
    X_pde = torch.cat([x_pde, y_pde, z_pde, t_pde], dim=1)
    
    # 2. Initial condition points (t=0) - uniformly distributed in space
    x_ic = torch.rand(n_ic, 1, device=device) * Lx
    y_ic = torch.rand(n_ic, 1, device=device) * Ly
    z_ic = torch.rand(n_ic, 1, device=device) * Lz
    t_ic = torch.zeros(n_ic, 1, device=device)
    X_ic = torch.cat([x_ic, y_ic, z_ic, t_ic], dim=1)
    
    # 3. Boundary condition points - evenly distributed across all faces
    n_per_face = n_bc // 6  # Six faces of the domain
    
    # Generate times for boundary points
    t_bc = torch.rand(n_per_face, 1, device=device) * t_max
    
    # Face 1: x=0
    x1 = torch.zeros(n_per_face, 1, device=device)
    y1 = torch.rand(n_per_face, 1, device=device) * Ly
    z1 = torch.rand(n_per_face, 1, device=device) * Lz
    bc1 = torch.cat([x1, y1, z1, t_bc.clone()], dim=1)
    
    # Face 2: x=Lx
    x2 = torch.ones(n_per_face, 1, device=device) * Lx
    y2 = torch.rand(n_per_face, 1, device=device) * Ly
    z2 = torch.rand(n_per_face, 1, device=device) * Lz
    bc2 = torch.cat([x2, y2, z2, t_bc.clone()], dim=1)
    
    # Face 3: y=0
    x3 = torch.rand(n_per_face, 1, device=device) * Lx
    y3 = torch.zeros(n_per_face, 1, device=device)
    z3 = torch.rand(n_per_face, 1, device=device) * Lz
    bc3 = torch.cat([x3, y3, z3, t_bc.clone()], dim=1)
    
    # Face 4: y=Ly
    x4 = torch.rand(n_per_face, 1, device=device) * Lx
    y4 = torch.ones(n_per_face, 1, device=device) * Ly
    z4 = torch.rand(n_per_face, 1, device=device) * Lz
    bc4 = torch.cat([x4, y4, z4, t_bc.clone()], dim=1)
    
    # Face 5: z=0
    x5 = torch.rand(n_per_face, 1, device=device) * Lx
    y5 = torch.rand(n_per_face, 1, device=device) * Ly
    z5 = torch.zeros(n_per_face, 1, device=device)
    bc5 = torch.cat([x5, y5, z5, t_bc.clone()], dim=1)
    
    # Face 6: z=Lz
    x6 = torch.rand(n_per_face, 1, device=device) * Lx
    y6 = torch.rand(n_per_face, 1, device=device) * Ly
    z6 = torch.ones(n_per_face, 1, device=device) * Lz
    bc6 = torch.cat([x6, y6, z6, t_bc.clone()], dim=1)
    
    # Combine all boundary points
    X_bc = torch.cat([bc1, bc2, bc3, bc4, bc5, bc6])
    
    return X_pde, X_ic, X_bc

def train_with_adam(model, physics, X_pde, X_ic, X_bc, optimizer, loss_weights, max_epochs, device, save_dir):
    """
    Train the PINN model using Adam optimizer.
    
    Args:
        model: PINN model
        physics: Physics module
        X_pde, X_ic, X_bc: Training points
        optimizer: Adam optimizer
        loss_weights: Dictionary of loss weights
        max_epochs: Maximum number of training epochs
        device: Computation device
        save_dir: Directory to save results
        
    Returns:
        loss_history: Dictionary of loss history
        best_loss: Best loss achieved
    """
    # Loss history
    loss_history = {
        'total': [],
        'pde': [],
        'ic': [],
        'bc': [],
    }
    
    # Best loss tracking
    best_loss = float('inf')
    
    # Timer
    start_time = time.time()
    last_save_time = start_time
    
    # Unpack coordinates
    x_pde, y_pde, z_pde, t_pde = X_pde[:, 0], X_pde[:, 1], X_pde[:, 2], X_pde[:, 3]
    x_ic, y_ic, z_ic, t_ic = X_ic[:, 0], X_ic[:, 1], X_ic[:, 2], X_ic[:, 3]
    x_bc, y_bc, z_bc, t_bc = X_bc[:, 0], X_bc[:, 1], X_bc[:, 2], X_bc[:, 3]
    
    print("Stage 1: Adam Optimization")
    for epoch in range(max_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # 1. PDE residual
        pde_residual = physics.pde_residual(model, x_pde, y_pde, z_pde, t_pde, epoch)
        loss_pde = torch.mean(pde_residual**2)
        
        # 2. Initial condition residual
        ic_residual = physics.ic_residual(model, x_ic, y_ic, z_ic, t_ic)
        loss_ic = torch.mean(ic_residual**2)
        
        # 3. Boundary condition residual
        bc_residual = physics.bc_residual(model, x_bc, y_bc, z_bc, t_bc)
        loss_bc = torch.mean(bc_residual**2)
        
        # Total loss with weights
        total_loss = (
            loss_weights['pde'] * loss_pde + 
            loss_weights['ic'] * loss_ic + 
            loss_weights['bc'] * loss_bc
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Record losses
        loss_history['total'].append(total_loss.item())
        loss_history['pde'].append(loss_pde.item())
        loss_history['ic'].append(loss_ic.item())
        loss_history['bc'].append(loss_bc.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{max_epochs} | "
                  f"Loss: {total_loss.item():.6e} | "
                  f"PDE: {loss_pde.item():.6e} | "
                  f"IC: {loss_ic.item():.6e} | "
                  f"BC: {loss_bc.item():.6e} | "
                  f"Time: {elapsed:.2f}s")
        
        # Save best model
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, os.path.join(save_dir, 'best_model.pt'))
        
        # Save checkpoint every 30 minutes
        current_time = time.time()
        if current_time - last_save_time > 1800:  # 1800 seconds = 30 minutes
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
                'loss_history': loss_history,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            last_save_time = current_time
    
    return loss_history, best_loss

def train_with_lbfgs(model, physics, X_pde, X_ic, X_bc, loss_weights, max_iter, device, save_dir, best_loss, loss_history):
    """
    Train the PINN model using L-BFGS optimizer.
    
    Args:
        model: PINN model
        physics: Physics module
        X_pde, X_ic, X_bc: Training points
        loss_weights: Dictionary of loss weights
        max_iter: Maximum number of L-BFGS iterations
        device: Computation device
        save_dir: Directory to save results
        best_loss: Best loss from Adam stage
        loss_history: Loss history from Adam stage
        
    Returns:
        loss_history: Updated dictionary of loss history
        best_loss: Updated best loss
    """
    # Unpack coordinates
    x_pde, y_pde, z_pde, t_pde = X_pde[:, 0], X_pde[:, 1], X_pde[:, 2], X_pde[:, 3]
    x_ic, y_ic, z_ic, t_ic = X_ic[:, 0], X_ic[:, 1], X_ic[:, 2], X_ic[:, 3]
    x_bc, y_bc, z_bc, t_bc = X_bc[:, 0], X_bc[:, 1], X_bc[:, 2], X_bc[:, 3]
    
    # Initialize L-BFGS optimizer
    optimizer_lbfgs = optim.LBFGS(model.parameters(), 
                                 lr=0.5,
                                 max_iter=100,
                                 max_eval=120,
                                 tolerance_grad=1e-7,
                                 tolerance_change=1e-9,
                                 history_size=50,
                                 line_search_fn='strong_wolfe')
    
    # Counter for L-BFGS iterations
    lbfgs_iter = [0]  # Use a list to allow modification in closure
    lbfgs_start_time = time.time()
    current_best_loss = best_loss
    
    def closure():
        nonlocal current_best_loss
        optimizer_lbfgs.zero_grad()
        
        # 1. PDE residual
        pde_residual = physics.pde_residual(model, x_pde, y_pde, z_pde, t_pde)
        loss_pde = torch.mean(pde_residual**2)
        
        # 2. Initial condition residual
        ic_residual = physics.ic_residual(model, x_ic, y_ic, z_ic, t_ic)
        loss_ic = torch.mean(ic_residual**2)
        
        # 3. Boundary condition residual
        bc_residual = physics.bc_residual(model, x_bc, y_bc, z_bc, t_bc)
        loss_bc = torch.mean(bc_residual**2)
        
        # Total loss with weights
        total_loss = (
            loss_weights['pde'] * loss_pde + 
            loss_weights['ic'] * loss_ic + 
            loss_weights['bc'] * loss_bc
        )
        
        # Backward pass
        total_loss.backward()
        
        # Record losses for L-BFGS iterations
        if lbfgs_iter[0] % 10 == 0:
            elapsed = time.time() - lbfgs_start_time
            print(f"L-BFGS Iter {lbfgs_iter[0]} | "
                  f"Loss: {total_loss.item():.6e} | "
                  f"PDE: {loss_pde.item():.6e} | "
                  f"IC: {loss_ic.item():.6e} | "
                  f"BC: {loss_bc.item():.6e} | "
                  f"Time: {elapsed:.2f}s")
            
            loss_history['total'].append(total_loss.item())
            loss_history['pde'].append(loss_pde.item())
            loss_history['ic'].append(loss_ic.item())
            loss_history['bc'].append(loss_bc.item())
            
            # Update best model if needed
            if total_loss.item() < current_best_loss:
                current_best_loss = total_loss.item()
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'loss': total_loss.item(),
                }, os.path.join(save_dir, 'best_model_lbfgs.pt'))
        
        lbfgs_iter[0] += 1
        
        return total_loss
    
    print("\nStage 2: L-BFGS Optimization")
    try:
        for _ in range(max_iter // 100):  # Run L-BFGS in chunks of 100 iterations
            optimizer_lbfgs.step(closure)
    except Exception as e:
        print(f"L-BFGS optimization failed: {e}. Using best model from Adam optimization.")
    
    return loss_history, best_loss

def train_pinn_LMD(model, physics, X_pde, X_ic, X_bc, optimizer, 
                  loss_weights, max_epochs, device, save_dir='results', lbfgs_epochs=100):
    """
    Train the PINN model for LMD simulation using two-stage optimization (Adam + L-BFGS).
    
    Args:
        model: PINN model
        physics: Physics module
        X_pde, X_ic, X_bc: Training points
        optimizer: Optimizer (Adam for first stage)
        loss_weights: Dictionary of loss weights
        max_epochs: Maximum number of training epochs for Adam
        device: Computation device
        save_dir: Directory to save results
        lbfgs_epochs: Number of L-BFGS iterations
        
    Returns:
        loss_history: Dictionary of loss history
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Stage 1: Adam Optimization
    loss_history, best_loss = train_with_adam(model, physics, X_pde, X_ic, X_bc, optimizer, 
                                              loss_weights, max_epochs, device, save_dir)
    
    # Stage 2: L-BFGS Optimization (if requested)
    if lbfgs_epochs > 0:
        loss_history, best_loss = train_with_lbfgs(model, physics, X_pde, X_ic, X_bc, 
                                                   loss_weights, lbfgs_epochs, device, save_dir, 
                                                   best_loss, loss_history)
    
    # Save final model
    torch.save({
        'epoch': max_epochs,
        'model_state_dict': model.state_dict(),
        'loss': best_loss,
        'loss_history': loss_history,
    }, os.path.join(save_dir, 'final_model.pt'))
    
    # Save loss history
    np.save(os.path.join(save_dir, 'loss_history.npy'), loss_history)
    
    # Print completion
    elapsed = time.time() - time.time()  # Note: This should be time.time() - start_time, but start_time is in train_with_adam
    print(f"Training completed. Best loss: {best_loss:.6e}")
    
    return loss_history

def plot_temperature_field(model, physics, time, save_path='results', plane='xy'):
    """
    Generate 2D contour plot of temperature field at a specific time
    Exactly matching the visualization in Figure 5 of the paper
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Increased resolution for better visualization (matching paper)
    n1, n2 = 200, 100
    
    # Create grid based on the selected plane
    if plane == 'xy':
        # Top surface (z = 0)
        coord1 = torch.linspace(0, physics.params['Lx'], n1, device=device)
        coord2 = torch.linspace(0, physics.params['Ly'], n2, device=device)
        X1, X2 = torch.meshgrid(coord1, coord2, indexing='ij')
        X3 = torch.zeros_like(X1)
        title = f'Temperature Field at t = {time:.2f}s (Top Surface, z = 0)'
        xlabel, ylabel = 'X (mm)', 'Y (mm)'
    elif plane == 'xz':
        # Center vertical slice (y = Ly/2)
        coord1 = torch.linspace(0, physics.params['Lx'], n1, device=device)
        coord2 = torch.linspace(0, physics.params['Lz'], n2, device=device)
        X1, X3 = torch.meshgrid(coord1, coord2, indexing='ij')
        X2 = torch.ones_like(X1) * physics.params['Ly'] / 2
        title = f'Temperature Field at t = {time:.2f}s (Center Slice, y = {physics.params["Ly"]/2*1000:.1f}mm)'
        xlabel, ylabel = 'X (mm)', 'Z (mm)'
    elif plane == 'yz':
        # Vertical slice at specific x position
        x_pos = physics.params['v'] * time  # At current laser position
        x_pos = min(x_pos, physics.params['Lx'])  # Ensure it's in the domain
        coord1 = torch.linspace(0, physics.params['Ly'], n1, device=device)
        coord2 = torch.linspace(0, physics.params['Lz'], n2, device=device)
        X2, X3 = torch.meshgrid(coord1, coord2, indexing='ij')
        X1 = torch.ones_like(X2) * x_pos
        title = f'Temperature Field at t = {time:.2f}s (Vertical Slice, x = {x_pos*1000:.1f}mm)'
        xlabel, ylabel = 'Y (mm)', 'Z (mm)'
    else:
        raise ValueError(f"Unsupported plane: {plane}")
    
    # Reshape for batch processing
    coord1_flat = X1.flatten()
    coord2_flat = X2.flatten()
    coord3_flat = X3.flatten()
    time_tensor = torch.ones_like(coord1_flat) * time
    
    # Stack coordinates (x, y, z, t)
    coords = torch.stack([coord1_flat, coord2_flat, coord3_flat, time_tensor], dim=1)
    with torch.no_grad():
        u_scaled = model(coords)
        temperature = physics.unscale_temperature(u_scaled)
        
        # Debug output to verify temperatures
        print(f"Debug - Temperature range: {temperature.min().item():.2f}K to {temperature.max().item():.2f}K")
    # Forward pass
    with torch.no_grad():
        u_scaled = model(coords)
        temperature = physics.unscale_temperature(u_scaled)
    
    # Reshape to grid
    temperature_grid = temperature.reshape(X1.shape).cpu().numpy()
    min_temp = temperature_grid.min()
    max_temp = temperature_grid.max()
    
    print(f"Debug - Plot temperature range: {min_temp:.2f}K to {max_temp:.2f}K")
    
    # Ensure we have distinct min and max values
    if np.isclose(min_temp, max_temp) or max_temp - min_temp < 1.0:
        print("Warning: Temperature range is too small, artificially expanding")
        max_temp = max(min_temp + 100.0, min_temp * 1.05)
    
    # Create levels that span from ambient to max predicted temperature
    levels = np.linspace(physics.params['T0'], max_temp, 100)
    # Convert coordinates to mm for plotting (matching paper units)
    X1_mm = X1.cpu().numpy() * 1000  # m to mm
    X2_mm = X2.cpu().numpy() * 1000  # m to mm
    
    # Create plot (similar to Figure 5 in paper)
    plt.figure(figsize=(12, 10))
    
    # Use 'hot' colormap with more levels as in the paper
    levels = np.linspace(physics.params['T0'], min(temperature_grid.max(), 3000), 100)
    contour = plt.contourf(X1_mm, X2_mm, temperature_grid, 
                          levels=levels, cmap='hot')
    
    # Add colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label('Temperature (K)', fontsize=12)
    
    # Add laser position marker if during deposition
    if time <= physics.params['t1'] and plane == 'xy':
        laser_pos_x = physics.params['v'] * time * 1000  # mm
        laser_pos_y = physics.params['Ly'] / 2 * 1000  # mm
        plt.plot(laser_pos_x, laser_pos_y, 'o', 
                markersize=10, color='cyan', label='Laser Position')
        
        # Add a circle to show the laser spot size
        laser_radius = physics.params['Ra'] * 1000  # mm
        circle = plt.Circle((laser_pos_x, laser_pos_y), laser_radius, 
                           fill=False, color='cyan', linestyle='--', linewidth=1.5)
        plt.gca().add_patch(circle)
        
        plt.legend(fontsize=12)
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(False)
    
    # Add melting point contour
    if 'Tm' in physics.params:
        melting_contour = plt.contour(X1_mm, X2_mm, temperature_grid, 
                   levels=[physics.params['Tm']], 
                   colors='cyan', linewidths=2, 
                   linestyles='dashed')
        plt.clabel(melting_contour, inline=True, fontsize=10, fmt='%1.0f K')
    
    # Add solidus temperature contour
    if 'Ts' in physics.params:
        solidus_contour = plt.contour(X1_mm, X2_mm, temperature_grid, 
                    levels=[physics.params['Ts']], 
                    colors='blue', linewidths=1.5, 
                    linestyles='dotted')
        plt.clabel(solidus_contour, inline=True, fontsize=10, fmt='%1.0f K')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'temperature_field_{plane}_t{time:.2f}.png'), dpi=300)
    plt.close()
    
    return temperature_grid

def plot_loss_history(loss_history, save_path='results'):
    """
    Plot the loss curves during training
    Matching Figure 4 in the paper
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Convert to numpy if needed
    if isinstance(loss_history, dict):
        for key in loss_history:
            if isinstance(loss_history[key], torch.Tensor):
                loss_history[key] = loss_history[key].cpu().numpy()
    elif isinstance(loss_history, str):
        # Load from file if path is provided
        loss_history = np.load(loss_history, allow_pickle=True).item()
    
    # Create x-axis (epochs)
    epochs = range(len(loss_history['total']))
    
    plt.figure(figsize=(12, 8))
    
    # Plot all loss components as in the paper
    plt.semilogy(epochs, loss_history['total'], label='Total Loss', linewidth=2)
    plt.semilogy(epochs, loss_history['pde'], label='PDE Loss', linewidth=2)
    plt.semilogy(epochs, loss_history['ic'], label='IC Loss', linewidth=2)
    plt.semilogy(epochs, loss_history['bc'], label='BC Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (Log Scale)', fontsize=14)
    plt.title('Training Loss History', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Format the y-axis with scientific notation
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatter(labelOnlyBase=False))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_history.png'), dpi=300)
    plt.savefig(os.path.join(save_path, 'loss_history.pdf'))
    plt.close()

def plot_temperature_distributions(model, physics, times, save_path='results'):
    """
    Plot temperature distribution along the laser path
    Matching Figure 6 in the paper
    
    Args:
        model: Trained PINN model
        physics: Physics module
        times: List of time points
        save_path: Directory to save figures
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Figure for temperature values
    plt.figure(figsize=(10, 6))
    
    # Define a color cycle for multiple lines
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    for i, t in enumerate(times):
        # Sample points along scanning track
        nx = 100
        x = torch.linspace(0, physics.params['Lx'], nx, device=device)
        y = torch.ones(nx, device=device) * (physics.params['Ly'] / 2)  # Center line
        z = torch.zeros(nx, device=device)  # Top surface
        t_tensor = torch.ones(nx, device=device) * t
        
        # Predict temperature
        inputs = torch.stack([x, y, z, t_tensor], dim=1)
        with torch.no_grad():
            u_scaled = model(inputs)
            temperatures = physics.unscale_temperature(u_scaled)
        
        # Convert to numpy for plotting
        x_np = x.cpu().numpy() * 1000  # m to mm
        temperatures_np = temperatures.cpu().numpy()
        
        # Plot temperature distribution
        plt.plot(x_np, temperatures_np, 
                 color=colors[i % len(colors)], 
                 label=f't={t:.1f}s', 
                 linewidth=2)
    
    # Complete and save temperature plot
    plt.xlabel('X Position (mm)', fontsize=14)
    plt.ylabel('Temperature (K)', fontsize=14)
    plt.title('Temperature Distribution Along Laser Path', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'temperature_along_path.png'), dpi=300)
    plt.close()

def evaluate_melt_pool_dimensions(model, physics, times):
    """
    Calculate melt pool dimensions at different times
    As shown in Figure 7 of the paper
    
    Args:
        model: Trained PINN model
        physics: Physics module
        times: Array of time points
        
    Returns:
        dimensions: Dictionary with melt pool lengths, widths, and depths
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Convert times to tensor if not already
    if not isinstance(times, torch.Tensor):
        times = torch.tensor(times, dtype=torch.float32, device=device)
    
    # Filter times to only include deposition phase
    valid_times = times[times <= physics.params['t1']]
    
    # Output containers
    lengths = []
    widths = []
    depths = []
    
    # Melting temperature
    Tm = physics.params['Tm']
    
    # For each valid time point
    for t in valid_times:
        # 1. Length (x-direction)
        nx = 200
        x = torch.linspace(0, physics.params['Lx'], nx, device=device)
        y = torch.ones_like(x) * (physics.params['Ly'] / 2)  # Center line
        z = torch.zeros_like(x)  # Top surface
        t_tensor = torch.ones_like(x) * t
        
        # Stack inputs
        inputs = torch.stack([x, y, z, t_tensor], dim=1)
        
        # Predict temperatures
        with torch.no_grad():
            u_scaled = model(inputs)
            temperatures = physics.unscale_temperature(u_scaled)
        
        # Find melt pool regions (T > Tm)
        is_molten = temperatures > Tm
        
        # Calculate length if molten region exists
        if is_molten.any():
            # Get indices of molten region
            molten_indices = torch.where(is_molten)[0]
            x_molten = x[molten_indices]
            length = (x_molten.max() - x_molten.min()).item() * 1000  # m to mm
        else:
            length = 0.0
        
        lengths.append(length)
        
        # 2. Width (y-direction)
        ny = 100
        y = torch.linspace(0, physics.params['Ly'], ny, device=device)
        x = torch.ones_like(y) * (physics.params['v'] * t)  # At laser position
        z = torch.zeros_like(y)  # Top surface
        t_tensor = torch.ones_like(y) * t
        
        # Stack inputs
        inputs = torch.stack([x, y, z, t_tensor], dim=1)
        
        # Predict temperatures
        with torch.no_grad():
            u_scaled = model(inputs)
            temperatures = physics.unscale_temperature(u_scaled)
        
        # Find melt pool regions (T > Tm)
        is_molten = temperatures > Tm
        
        # Calculate width if molten region exists
        if is_molten.any():
            # Get indices of molten region
            molten_indices = torch.where(is_molten)[0]
            y_molten = y[molten_indices]
            width = (y_molten.max() - y_molten.min()).item() * 1000  # m to mm
        else:
            width = 0.0
        
        widths.append(width)
        
        # 3. Depth (z-direction)
        nz = 50
        z = torch.linspace(0, min(0.002, physics.params['Lz']), nz, device=device)  # Focus on shallow depth
        x = torch.ones_like(z) * (physics.params['v'] * t)  # At laser position
        y = torch.ones_like(z) * (physics.params['Ly'] / 2)  # Center line
        t_tensor = torch.ones_like(z) * t
        
        # Stack inputs
        inputs = torch.stack([x, y, z, t_tensor], dim=1)
        
        # Predict temperatures
        with torch.no_grad():
            u_scaled = model(inputs)
            temperatures = physics.unscale_temperature(u_scaled)
        
        # Find melt pool regions (T > Tm)
        is_molten = temperatures > Tm
        
        # Calculate depth if molten region exists
        if is_molten.any():
            # Get maximum depth of molten region
            z_molten = z[is_molten]
            depth = z_molten.max().item() * 1000  # m to mm
        else:
            depth = 0.0
        
        depths.append(depth)
    
    return {
        'times': valid_times.cpu().numpy(),
        'lengths': np.array(lengths),
        'widths': np.array(widths),
        'depths': np.array(depths)
    }

def plot_melt_pool_evolution(model, physics, times, save_path='results'):
    """
    Plot melt pool dimensions over time
    Matching Figure 7 in the paper
    
    Args:
        model: Trained PINN model
        physics: Physics module
        times: Array of time points
        save_path: Directory to save the figure
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Calculate melt pool dimensions
    dimensions = evaluate_melt_pool_dimensions(model, physics, times)
    
    # Plot dimensions
    plt.figure(figsize=(12, 8))
    
    plt.plot(dimensions['times'], dimensions['lengths'], 'ro-', label='Length', linewidth=2)
    plt.plot(dimensions['times'], dimensions['widths'], 'go-', label='Width', linewidth=2)
    plt.plot(dimensions['times'], dimensions['depths'], 'bo-', label='Depth', linewidth=2)
    
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Melt Pool Dimension (mm)', fontsize=14)
    plt.title('Melt Pool Evolution During Deposition', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'melt_pool_evolution.png'), dpi=300)
    plt.close()
    
    return dimensions

def implement_transfer_learning(base_model, new_params, device, 
                               n_pde=5000, n_ic=2000, n_bc=2000, 
                               max_epochs=2000, save_dir='transfer_results'):
    """
    Implement transfer learning as shown in Section 5.3 of the paper
    
    Args:
        base_model: Pre-trained PINN model
        new_params: New manufacturing parameters
        device: Computation device
        n_pde, n_ic, n_bc: Number of training points
        max_epochs: Maximum number of training epochs
        save_dir: Directory to save results
        
    Returns:
        new_model: Transferred model
        new_physics: Physics module with new parameters
    """
    # Create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create new physics module with new parameters
    new_physics = Physics(new_params)
    
    # Create new model with same architecture
    new_model = PINN(
        input_dim=4,
        hidden_dim=30,
        output_dim=1
    ).to(device)
    
    # Copy parameters from base model (transfer learning)
    new_model.load_state_dict(base_model.state_dict())
    
    # Generate training points for new conditions
    X_pde, X_ic, X_bc = generate_training_points(new_params, n_pde, n_ic, n_bc, device)
    
    # Initialize optimizer (lower learning rate for transfer learning)
    optimizer = optim.Adam(new_model.parameters(), lr=5e-5)
    
    # Define loss weights (same as base training)
    loss_weights = {
        'pde': 5.0,
        'ic': 1.0,
        'bc': 10.0
    }
    
    # Train with transfer learning (fewer epochs needed)
    print(f"Starting transfer learning for {max_epochs} epochs...")
    loss_history = train_pinn_LMD(
        new_model, new_physics, X_pde, X_ic, X_bc, optimizer, 
        loss_weights, max_epochs, device, save_dir=save_dir
    )
    
    # Plot loss history for transfer learning
    plot_loss_history(loss_history, save_dir)
    
    return new_model, new_physics

def main():
    """
    Main function to run the PINN-LMD simulation
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Physics-Informed Neural Network for Laser Metal Deposition')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'plot', 'transfer'],
                       help='Mode to run (train, evaluate, plot, transfer)')
    parser.add_argument('--epochs', type=int, default=3000,
                       help='Number of training epochs (default: 10000)')
    parser.add_argument('--lbfgs_epochs', type=int, default=16000,
                       help='Number of L-BFGS epochs (default: 100, 0 to skip)')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to load pretrained model (for evaluate/plot/transfer modes)')
    parser.add_argument('--plot_times', type=str, default='0.5,1.0,1.5,2.0',
                       help='Comma-separated list of time points for plotting')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--laser_power', type=float, default=2000.0,
                       help='Laser power in watts (default: 2000W from paper)')
    parser.add_argument('--fast', action='store_true',
                       help='Use reduced training points for faster execution')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU device {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Problem parameters based on the paper (Table 2)
    params = {
        'rho': 7780.0,                # Density (kg/m^3)
        'eta': 0.75,                  # Laser absorption coefficient
        'P': args.laser_power,        # Laser power (W)
        'v': 8.0e-3,                  # Scanning speed (m/s)
        'Ra': 3.0e-3,                 # Laser spot radius in x (m)
        'Rb': 3.0e-3,                 # Laser spot radius in y (m)
        'Rc': 1.0e-3,                 # Laser spot radius in z (m)
        'h': 20.0,                    # Convection coefficient (W/(m^2·K))
        'epsilon': 0.85,              # Emissivity
        'T0': 293.15,                 # Initial/ambient temperature (K)
        'Tm': 1730.0,                 # Melting temperature (K)
        'Ts': 1690.0,                 # Solidus temperature (K)
        'Lx': 0.04,                   # Domain length in x (m)
        'Ly': 0.02,                   # Domain width in y (m)
        'Lz': 0.005,                  # Domain height in z (m)
        't1': 2.0,                    # Deposition time (s)
        't2': 10.0                    # Cooling time (s)
    }
    
    # Initialize physics module
    physics = Physics(params)
    
    # Initialize or load model
    if args.mode in ['evaluate', 'plot', 'transfer'] and args.model_path:
        # Load pretrained model
        print(f"Loading model from {args.model_path}")
        try:
            model = PINN(input_dim=4, hidden_dim=30, output_dim=1).to(device)
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"Model loaded with loss: {checkpoint.get('loss', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        # Create new model for training (30 neurons as per paper)
        model = PINN(input_dim=4, hidden_dim=30, output_dim=1).to(device)
        
        # Calculate input normalization stats
        input_mean = torch.tensor([params['Lx']/2, params['Ly']/2, params['Lz']/2, params['t1']/2])
        input_std = torch.tensor([params['Lx'], params['Ly'], params['Lz'], params['t1'] + params['t2']])
        model.set_input_stats(input_mean, input_std)
        
        print(f"Created model with 30 hidden neurons and 1 residual block")
    
    # Run in the specified mode
    if args.mode == 'train':
        # Determine number of training points
        if args.fast:
            # Reduced points for faster training
            n_pde = 7000
            n_ic = 5000
            n_bc = 5000
        else:
            # Full points as in the paper (30,000 points)
            n_pde = 20000
            n_ic = 30000 
            n_bc = 10000
        
        print(f"Generating training points: {n_pde} PDE, {n_ic} IC, {n_bc} BC")
        X_pde, X_ic, X_bc = generate_training_points(params, n_pde, n_ic, n_bc, device)
        
        # Define loss weights as in the paper
        loss_weights = {
            'pde': 10.0,   # Higher weight on PDE residual
            'ic': 1.0,    # Standard weight on initial conditions
            'bc': 1.0    # Higher weight on boundary conditions
        }
        
        # Initialize Adam optimizer (first stage)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Train model (two-stage optimization)
        print(f"Starting training for {args.epochs} epochs (Adam) + {args.lbfgs_epochs} iterations (L-BFGS)...")
        loss_history = train_pinn_LMD(
            model, physics, X_pde, X_ic, X_bc, optimizer, 
            loss_weights, args.epochs, device, save_dir=args.save_dir, lbfgs_epochs=args.lbfgs_epochs
        )
        
        # Plot loss history
        plot_loss_history(loss_history, args.save_dir)
        
        # Generate initial temperature field plots
        try:
            plot_temperature_field(model, physics, 0.5, args.save_dir, plane='xy')
            plot_temperature_field(model, physics, 1.0, args.save_dir, plane='xy')
            plot_temperature_field(model, physics, 2.0, args.save_dir, plane='xy')
        except Exception as e:
            print(f"Error plotting temperature fields: {e}")
    
    elif args.mode == 'evaluate':
        # Parse plot times
        plot_times = [float(t) for t in args.plot_times.split(',')]
        
        # Evaluate model at specified times
        for t in plot_times:
            try:
                plot_temperature_field(model, physics, t, args.save_dir, plane='xy')
                plot_temperature_field(model, physics, t, args.save_dir, plane='xz')
            except Exception as e:
                print(f"Error plotting at time {t}: {e}")
        
        # Plot temperature distributions along scanning track
        try:
            plot_temperature_distributions(model, physics, plot_times, args.save_dir)
        except Exception as e:
            print(f"Error plotting temperature distributions: {e}")
        
        # Melt pool evolution during deposition
        try:
            device = next(model.parameters()).device
            t_mp = torch.linspace(0.1, params['t1'], 10, device=device)
            plot_melt_pool_evolution(model, physics, t_mp, args.save_dir)
        except Exception as e:
            print(f"Error plotting melt pool evolution: {e}")
    
    elif args.mode == 'plot':
        # Parse plot times
        plot_times = [float(t) for t in args.plot_times.split(',')]
        
        # Generate plots at specified times
        for t in plot_times:
            try:
                plot_temperature_field(model, physics, t, args.save_dir, plane='xy')
                plot_temperature_field(model, physics, t, args.save_dir, plane='xz')
                plot_temperature_field(model, physics, t, args.save_dir, plane='yz')
            except Exception as e:
                print(f"Error plotting at time {t}: {e}")
    
    elif args.mode == 'transfer':
        # Define new parameters for transfer learning (different laser power)
        new_params = params.copy()
        new_params['P'] = 2500.0  # Higher laser power
        
        # Implement transfer learning
        transfer_save_dir = os.path.join(args.save_dir, 'transfer')
        new_model, new_physics = implement_transfer_learning(
            model, new_params, device, 
            max_epochs=args.epochs // 2,  # Fewer epochs needed for transfer learning
            save_dir=transfer_save_dir
        )
        
        # Generate comparison plots for original and transfer models
        try:
            # Plot temperature fields for both models at t=1.0s
            t_eval = 1.0  # Evaluation time
            plot_temperature_field(model, physics, t_eval, args.save_dir, plane='xy')
            plot_temperature_field(new_model, new_physics, t_eval, transfer_save_dir, plane='xy')
            
            # Compare melt pool dimensions
            t_mp = torch.linspace(0.1, min(params['t1'], new_params['t1']), 10, device=device)
            original_mp = evaluate_melt_pool_dimensions(model, physics, t_mp)
            transfer_mp = evaluate_melt_pool_dimensions(new_model, new_physics, t_mp)
            
            # Plot comparison of melt pool dimensions
            plt.figure(figsize=(12, 8))
            
            plt.plot(original_mp['times'], original_mp['lengths'], 'ro-', label='Length (Original)', linewidth=2)
            plt.plot(transfer_mp['times'], transfer_mp['lengths'], 'ro--', label='Length (Transfer)', linewidth=2)
            
            plt.plot(original_mp['times'], original_mp['widths'], 'go-', label='Width (Original)', linewidth=2)
            plt.plot(transfer_mp['times'], transfer_mp['widths'], 'go--', label='Width (Transfer)', linewidth=2)
            
            plt.plot(original_mp['times'], original_mp['depths'], 'bo-', label='Depth (Original)', linewidth=2)
            plt.plot(transfer_mp['times'], transfer_mp['depths'], 'bo--', label='Depth (Transfer)', linewidth=2)
            
            plt.xlabel('Time (s)', fontsize=14)
            plt.ylabel('Melt Pool Dimension (mm)', fontsize=14)
            plt.title('Melt Pool Comparison: Original vs Transfer Learning', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(transfer_save_dir, 'melt_pool_comparison.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error in transfer learning evaluation: {e}")
    
    print(f"All results saved to {args.save_dir}")

if __name__ == "__main__":
    main()
