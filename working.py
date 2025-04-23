import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Use double precision for better numerical stability
torch.set_default_dtype(torch.float64)

# ---- Adaptive Activation ----
class AdaptiveTanh(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1) * n)
        
    def forward(self, x):
        return torch.tanh(self.a * x)

# ---- Residual Block ----
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = AdaptiveTanh()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)
        
    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        return out + identity

# ---- PINN Model ----
class PINN_LMD(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=50, output_dim=1, params=None):
        super().__init__()
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('k_x', torch.ones(1))
        self.register_buffer('k_t', torch.ones(1))
        self.register_buffer('k_u', torch.ones(1))
        self.params = params
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.res_block = ResidualBlock(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_normal_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)
        
    def set_scaling_factors(self, Lx, Ly, Lz, t1, t2, T_max):
        self.k_x.data = torch.tensor(1.0 / max(Lx, Ly, Lz))
        self.k_t.data = torch.tensor(1.0 / (t1 + t2))
        self.k_u.data = torch.tensor(1.0 / T_max)
        
    def set_input_stats(self, mean, std):
        self.input_mean.data.copy_(torch.as_tensor(mean, dtype=torch.get_default_dtype()))
        self.input_std.data.copy_(torch.as_tensor(std, dtype=torch.get_default_dtype()))
        
    def forward(self, x):
        x_norm = (x - self.input_mean) / self.input_std
        x_scaled = torch.cat([x_norm[:, :3] * self.k_x, x_norm[:, 3:] * self.k_t], dim=1)
        
        h = torch.tanh(self.input_layer(x_scaled))
        h = torch.tanh(self.hidden1(h))
        h = self.res_block(h)
        h = torch.tanh(h)
        
        # Use softplus with a smaller multiplier to prevent explosive growth
        u_scaled = 0.5 * nn.functional.softplus(self.output_layer(h))
        
        # Apply proper scaling - add ambient temperature base
        T_ambient = self.params['T0']
        
        # Add explicit temperature clamping to avoid unrealistic temperatures
        T_melting = 1930.0  # Approximate melting point for Ti6Al4V
        T_max_allowed = 3500.0  # Maximum physically plausible temperature
        
        unclamped_T = T_ambient + u_scaled / self.k_u
        return torch.clamp(unclamped_T, min=T_ambient, max=T_max_allowed)

# ---- Physics Module ----
class LMDPhysics:
    def __init__(self, params):
        self.params = params
        self.sigma_sb = 5.67e-8
        self.Lx, self.Ly, self.Lz = params['Lx'], params['Ly'], params['Lz']
        self.t1, self.t2 = params['t1'], params['t2']
        self.Q_max = 1.0e6  # Normalize heat source for numerical stability
        self.T_ref = 4000.0  # Increased reference temperature for scaling
        
    def thermal_conductivity(self, T):
        # Improved Ti6Al4V thermal conductivity model
        # Values taken from literature for Ti6Al4V
        k_room = 6.7  # at room temp
        k_high = 33.4  # at high temp
        
        # Linear interpolation between two reference points
        k = k_room + (k_high - k_room) * (T - 300) / (1900 - 300)
        
        # Ensure reasonable bounds
        return torch.clamp(k, min=6.0, max=40.0)
        
    def specific_heat(self, T):
        # Improved Ti6Al4V specific heat model
        # Corrected from previous implementation which had a copy-paste error
        
        # Values for Ti6Al4V
        cp_room = 560.0  # J/(kg·K) at room temperature
        cp_high = 700.0  # J/(kg·K) at high temperature
        
        # Linear interpolation
        cp = cp_room + (cp_high - cp_room) * (T - 300) / (1900 - 300)
        
        # Add latent heat effect near melting point (simplified)
        T_melt = 1930.0
        melt_range = 100.0
        melt_factor = torch.exp(-(T - T_melt)**2 / (2 * melt_range**2))
        cp_boost = 500.0 * melt_factor  # Additional heat capacity near melting
        
        return torch.clamp(cp + cp_boost, min=500.0, max=1500.0)
        
    def laser_heat_source(self, x, y, z, t):
        v, P = self.params['v'], self.params['P']
        eta, Ra, Rb, Rc = self.params['eta'], self.params['Ra'], self.params['Rb'], self.params['Rc']
        x0, y0, z0 = 0.0, self.Ly / 2, 0.0
        
        # Add a small smoothing factor to improve stability
        active = (t <= self.t1).float()
        
        # Use a slightly larger spot size for improved stability
        Ra_eff = Ra * 1.1
        Rb_eff = Rb * 1.1
        Rc_eff = Rc * 1.1
        
        r_sq = ((x - (v * t + x0)) / Ra_eff)**2 + ((y - y0) / Rb_eff)**2 + ((z - z0) / Rc_eff)**2
        
        # Adjust power scaling for improved numerical stability
        Q = (6 * np.sqrt(3) * eta * P / (np.pi * np.sqrt(np.pi) * Ra_eff * Rb_eff * Rc_eff)) * torch.exp(-3 * r_sq) * active
        
        return Q / self.Q_max
        
    def compute_derivatives(self, model, x, y, z, t):
        # Ensure all inputs have the same size
        min_size = min(x.size(0), y.size(0), z.size(0), t.size(0))
        x = x[:min_size]
        y = y[:min_size]
        z = z[:min_size]
        t = t[:min_size]
        
        coords = [x.clone().requires_grad_(True), 
                  y.clone().requires_grad_(True),
                  z.clone().requires_grad_(True), 
                  t.clone().requires_grad_(True)]
                  
        inputs = torch.stack(coords, dim=1)
        T = model(inputs)
        
        # Check for extremely high temperatures that indicate potential issues
        with torch.no_grad():
            max_temp = T.max().item()
            if max_temp > 3500:
                print(f"⚠️ High temperature detected: {max_temp:.1f}K — model may need adjustment.")
        
        grad_outputs = torch.ones_like(T)
        dT_dx, dT_dy, dT_dz, dT_dt = grad(T, coords, grad_outputs=grad_outputs, create_graph=True)
        
        d2T_dx2 = grad(dT_dx, coords[0], grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0]
        d2T_dy2 = grad(dT_dy, coords[1], grad_outputs=torch.ones_like(dT_dy), create_graph=True)[0]
        d2T_dz2 = grad(dT_dz, coords[2], grad_outputs=torch.ones_like(dT_dz), create_graph=True)[0]
        
        return T, dT_dt, dT_dx, dT_dy, dT_dz, d2T_dx2, d2T_dy2, d2T_dz2
        
    def pde_residual(self, model, x, y, z, t):
        T, dT_dt, dT_dx, dT_dy, dT_dz, d2T_dx2, d2T_dy2, d2T_dz2 = self.compute_derivatives(model, x, y, z, t)
        
        rho = self.params['rho']
        Cp = self.specific_heat(T)
        k = self.thermal_conductivity(T)
        Q = self.laser_heat_source(x, y, z, t) * self.Q_max  # Scale back the source term
        
        # Adjusted scale factor for better numerical stability
        scale_factor = 1.0e7
        
        # Add small regularization to prevent exploding gradients
        regularization = 1e-6 * (dT_dx**2 + dT_dy**2 + dT_dz**2)
        
        residual = rho * Cp * dT_dt - k * (d2T_dx2 + d2T_dy2 + d2T_dz2) - Q + regularization
        
        return residual / scale_factor
        
    def ic_residual(self, model, x, y, z, t):
        # Ensure all inputs have the same size
        min_size = min(x.size(0), y.size(0), z.size(0), t.size(0))
        x = x[:min_size]
        y = y[:min_size]
        z = z[:min_size]
        t = t[:min_size]
        
        inputs = torch.stack([x, y, z, t], dim=1)
        T = model(inputs)
        return (T - self.params['T0']) / self.T_ref
        
    def bc_residual(self, model, x, y, z, t):
        # Ensure all inputs have the same size
        min_size = min(x.size(0), y.size(0), z.size(0), t.size(0))
        x = x[:min_size]
        y = y[:min_size]
        z = z[:min_size]
        t = t[:min_size]
        
        T, _, dT_dx, dT_dy, dT_dz, _, _, _ = self.compute_derivatives(model, x, y, z, t)
        
        tol = 1e-4
        is_x_min = torch.abs(x) < tol
        is_x_max = torch.abs(x - self.Lx) < tol
        is_y_min = torch.abs(y) < tol
        is_y_max = torch.abs(y - self.Ly) < tol
        is_z_min = torch.abs(z) < tol
        is_z_max = torch.abs(z - self.Lz) < tol
        
        bc_residual = torch.zeros_like(T)
        k = self.thermal_conductivity(T)
        h = self.params['h']
        epsilon = self.params['epsilon']
        T0 = self.params['T0']
        
        q_conv = h * (T - T0)
        q_rad = epsilon * self.sigma_sb * (T**4 - T0**4)
        
        # Directional adjustment for boundaries
        bc_residual = torch.where(is_x_min, -k * dT_dx + q_conv + q_rad, bc_residual)
        bc_residual = torch.where(is_x_max, k * dT_dx + q_conv + q_rad, bc_residual)
        bc_residual = torch.where(is_y_min, -k * dT_dy + q_conv + q_rad, bc_residual)
        bc_residual = torch.where(is_y_max, k * dT_dy + q_conv + q_rad, bc_residual)
        bc_residual = torch.where(is_z_min, -k * dT_dz + q_conv + q_rad, bc_residual)
        bc_residual = torch.where(is_z_max, k * dT_dz + q_conv + q_rad, bc_residual)
        
        return bc_residual / self.Q_max

# ---- Training Module ----
class PINNTrainer:
    def __init__(self, model, physics, params):
        self.model = model
        self.physics = physics
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Set scaling factors - increased T_max to accommodate higher temperatures
        self.model.set_scaling_factors(
            params['Lx'], params['Ly'], params['Lz'], 
            params['t1'], params['t2'], 
            5000.0  # Increased T_max for normalization
        )
        
        # Learning rate scheduler and optimizer - reduced learning rate
        self.optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        
        # Loss weights
        self.pde_weight = torch.tensor(1.0, device=self.device, dtype=torch.get_default_dtype())
        self.bc_weight = torch.tensor(1.0, device=self.device, dtype=torch.get_default_dtype())
        self.ic_weight = torch.tensor(1.0, device=self.device, dtype=torch.get_default_dtype())
    
    def sample_collocation_points(self, n_pde, n_ic, n_bc):
        # Ensure n_bc is divisible by 6 (for even distribution across boundaries)
        n_bc_per_side = n_bc // 6
        n_bc = n_bc_per_side * 6  # Adjust total to ensure perfect division
        
        # Domain points for PDE residual - use stratified sampling for better coverage
        # Generate more points near the laser path
        laser_y = self.params['Ly'] / 2
        laser_z_min = 0
        laser_z_max = self.params['Ra'] * 3  # Sample more densely around the laser penetration zone
        
        # 80% of points in the area of interest
        n_pde_focus = int(0.8 * n_pde)
        n_pde_global = n_pde - n_pde_focus
        
        # Global points
        x_pde_global = torch.rand(n_pde_global, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lx']
        y_pde_global = torch.rand(n_pde_global, device=self.device, dtype=torch.get_default_dtype()) * self.params['Ly']
        z_pde_global = torch.rand(n_pde_global, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lz']
        
        # Focused points around laser path
        x_pde_focus = torch.rand(n_pde_focus, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lx']
        y_focus_width = self.params['Rb'] * 5
        y_pde_focus = laser_y + (torch.rand(n_pde_focus, device=self.device, dtype=torch.get_default_dtype()) * 2 - 1) * y_focus_width
        y_pde_focus = torch.clamp(y_pde_focus, 0, self.params['Ly'])
        z_pde_focus = torch.rand(n_pde_focus, device=self.device, dtype=torch.get_default_dtype()) * laser_z_max
        
        # Combine global and focused points
        x_pde = torch.cat([x_pde_global, x_pde_focus])
        y_pde = torch.cat([y_pde_global, y_pde_focus])
        z_pde = torch.cat([z_pde_global, z_pde_focus])
        
        # Use stratified time sampling to ensure coverage of all stages
        t_segments = 10
        t_points_per_segment = n_pde // t_segments
        t_pde = torch.zeros(n_pde, device=self.device, dtype=torch.get_default_dtype())
        
        for i in range(t_segments):
            t_start = i * (self.params['t1'] + self.params['t2']) / t_segments
            t_end = (i+1) * (self.params['t1'] + self.params['t2']) / t_segments
            t_segment = t_start + torch.rand(t_points_per_segment, device=self.device, dtype=torch.get_default_dtype()) * (t_end - t_start)
            t_pde[i*t_points_per_segment:(i+1)*t_points_per_segment] = t_segment
        
        # Initial condition points
        x_ic = torch.rand(n_ic, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lx']
        y_ic = torch.rand(n_ic, device=self.device, dtype=torch.get_default_dtype()) * self.params['Ly']
        z_ic = torch.rand(n_ic, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lz']
        t_ic = torch.zeros(n_ic, device=self.device, dtype=torch.get_default_dtype())
        
        # Boundary points - exactly n_bc_per_side points for each side
        
        # x = 0 and x = Lx sides
        x_bc_x0 = torch.zeros(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype())
        y_bc_x0 = torch.rand(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Ly']
        z_bc_x0 = torch.rand(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lz']
        
        x_bc_xL = torch.ones(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lx']
        y_bc_xL = torch.rand(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Ly']
        z_bc_xL = torch.rand(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lz']
        
        # y = 0 and y = Ly sides
        x_bc_y0 = torch.rand(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lx']
        y_bc_y0 = torch.zeros(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype())
        z_bc_y0 = torch.rand(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lz']
        
        x_bc_yL = torch.rand(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lx']
        y_bc_yL = torch.ones(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Ly']
        z_bc_yL = torch.rand(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lz']
        
        # z = 0 and z = Lz sides
        x_bc_z0 = torch.rand(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lx']
        y_bc_z0 = torch.rand(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Ly']
        z_bc_z0 = torch.zeros(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype())
        
        x_bc_zL = torch.rand(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lx']
        y_bc_zL = torch.rand(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Ly']
        z_bc_zL = torch.ones(n_bc_per_side, device=self.device, dtype=torch.get_default_dtype()) * self.params['Lz']
        
        # Combine all boundary points
        x_bc = torch.cat([x_bc_x0, x_bc_xL, x_bc_y0, x_bc_yL, x_bc_z0, x_bc_zL])
        y_bc = torch.cat([y_bc_x0, y_bc_xL, y_bc_y0, y_bc_yL, y_bc_z0, y_bc_zL])
        z_bc = torch.cat([z_bc_x0, z_bc_xL, z_bc_y0, z_bc_yL, z_bc_z0, z_bc_zL])
        t_bc = torch.rand(n_bc, device=self.device, dtype=torch.get_default_dtype()) * (self.params['t1'] + self.params['t2'])
        
        # Verify all tensors have consistent shapes
        assert x_bc.shape == t_bc.shape, f"Boundary x and t shapes don't match: {x_bc.shape} vs {t_bc.shape}"
        
        return {
            'pde': (x_pde, y_pde, z_pde, t_pde),
            'ic': (x_ic, y_ic, z_ic, t_ic),
            'bc': (x_bc, y_bc, z_bc, t_bc)
        }
    
    def update_loss_weights(self):
        # Self-adaptive weights based on loss values
        with torch.no_grad():
            weights_sum = self.pde_weight + self.bc_weight + self.ic_weight
            self.pde_weight = weights_sum / (self.pde_weight + 1e-10)
            self.bc_weight = weights_sum / (self.bc_weight + 1e-10)
            self.ic_weight = weights_sum / (self.ic_weight + 1e-10)
    
    def train(self, epochs, n_pde, n_ic, n_bc, weight_update_freq=1000):
        history = {'total': [], 'pde': [], 'ic': [], 'bc': [], 'max_temp': []}
        
        with tqdm(total=epochs, desc="Training Progress") as pbar:
            for epoch in range(epochs):
                # Sample points
                points = self.sample_collocation_points(n_pde, n_ic, n_bc)
                
                # PDE residual
                pde_residual = self.physics.pde_residual(
                    self.model, 
                    *points['pde']
                )
                pde_loss = torch.mean(pde_residual**2)
                
                # IC residual
                ic_residual = self.physics.ic_residual(
                    self.model,
                    *points['ic']
                )
                ic_loss = torch.mean(ic_residual**2)
                
                # BC residual
                bc_residual = self.physics.bc_residual(
                    self.model,
                    *points['bc']
                )
                bc_loss = torch.mean(bc_residual**2)
                
                # Update weights periodically
                if epoch % weight_update_freq == 0 and epoch > 0:
                    self.update_loss_weights()
                
                # Total loss
                total_loss = (
                    self.pde_weight * pde_loss + 
                    self.ic_weight * ic_loss + 
                    self.bc_weight * bc_loss
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping to prevent exploding gradients - reduced max_norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Monitor max temperature (for debugging)
                if epoch % 100 == 0:
                    with torch.no_grad():
                        test_inputs = torch.stack(points['pde'], dim=1)
                        pred_T = self.model(test_inputs)
                        max_temp = pred_T.max().item()
                        history['max_temp'].append(max_temp)
                        
                        # Print max temperature for monitoring
                        if epoch % 500 == 0:
                            print(f"[Epoch {epoch}] Max predicted T: {max_temp:.2f} K")
                
                # Store losses
                history['total'].append(total_loss.item())
                history['pde'].append(pde_loss.item())
                history['ic'].append(ic_loss.item())
                history['bc'].append(bc_loss.item())
                
                # Update progress bar
                pbar.update(1)
                if epoch % 100 == 0:
                    pbar.set_postfix({
                        'loss': total_loss.item(), 
                        'pde': pde_loss.item(),
                        'ic': ic_loss.item(),
                        'bc': bc_loss.item(),
                        'max_T': max_temp if 'max_temp' in locals() else 'N/A'
                    })
                
        return history

# ---- Plotting Functions ----
def plot_loss(history):
    """Plot the loss history"""
    plt.figure(figsize=(12, 8))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.semilogy(history['total'], label='Total Loss')
    plt.semilogy(history['pde'], label='PDE Loss')
    plt.semilogy(history['ic'], label='IC Loss')
    plt.semilogy(history['bc'], label='BC Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('PINN Training Loss History')
    plt.grid(True)
    
    # Plot max temperature when available
    if 'max_temp' in history and len(history['max_temp']) > 0:
        plt.subplot(2, 1, 2)
        plt.plot(range(0, len(history['max_temp'])*100, 100), history['max_temp'])
        plt.xlabel('Epochs')
        plt.ylabel('Max Temperature (K)')
        plt.title('Maximum Temperature Evolution')
        plt.grid(True)
        plt.axhline(y=3500, color='r', linestyle='--', label='Temperature Limit')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('loss_history.png', dpi=300)
    plt.close()

def plot_temperature_field(model, params, time_step, z_value=0.0):
    """Plot a 2D temperature field at specific z-height and time"""
    device = next(model.parameters()).device
    
    # Create mesh grid for 2D plot
    x = torch.linspace(0, params['Lx'], 100, device=device, dtype=torch.get_default_dtype())
    y = torch.linspace(0, params['Ly'], 100, device=device, dtype=torch.get_default_dtype())
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten the coordinates
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)
    Z_flat = torch.ones_like(X_flat) * z_value
    T_flat = torch.ones_like(X_flat) * time_step
    
    # Prepare input
    inputs = torch.stack([X_flat, Y_flat, Z_flat, T_flat], dim=1)
    
    # Get temperature predictions
    with torch.no_grad():
        T_pred = model(inputs)
    
    # Reshape for plotting
    T_grid = T_pred.reshape(X.shape)
    
    # Convert to numpy for plotting
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    T_np = T_grid.cpu().numpy()
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X_np, Y_np, T_np, 50, cmap='hot')
    plt.colorbar(contour, label='Temperature (K)')
    
    # Add laser path and position
    if time_step <= params['t1']:  # Only show during deposition
        laser_y = params['Ly'] / 2  # Middle of y-axis
        laser_x = params['v'] * time_step  # Position based on velocity
        
        # Add laser path
        plt.plot([0, params['Lx']], [laser_y, laser_y], 'w--', label='Laser path')
        
        # Add current laser position
        plt.scatter([laser_x], [laser_y], color='white', s=100, label='Current laser position')
        plt.legend(loc='upper right')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Temperature Field at z={z_value:.1f}, t={time_step:.1f}s')
    plt.tight_layout()
    plt.savefig(f'temp_field_z{z_value:.1f}_t{time_step:.1f}.png', dpi=300)
    plt.close()
    
    return T_np

# ---- Main Function ----
def main():
    # Set parameters for LMD simulation
    params = {
        'Lx': 0.040,          # Domain length in x-direction (m)
        'Ly': 0.020,          # Domain width in y-direction (m)
        'Lz': 0.005,          # Domain height in z-direction (m)
        't1': 2.0,            # Deposition time (s)
        't2': 1.0,            # Cooling time (s)
        'rho': 7780.0,        # Density (kg/m^3)
        'h': 20.0,            # Convection coefficient (W/m^2-K)
        'epsilon': 0.85,      # Emissivity
        'v': 0.008,           # Scanning speed (m/s)
        'P': 1800.0,          # Laser power (W) - slightly reduced from 2000W
        'eta': 0.7,           # Laser absorption coefficient - reduced from 0.75
        'Ra': 0.0035,         # Laser spot radius in x (m) - increased from 0.003
        'Rb': 0.0035,         # Laser spot radius in y (m) - increased from 0.003
        'Rc': 0.001,          # Laser penetration depth (m)
        'T0': 293.15,         # Initial/ambient temperature (K)
        'learning_rate': 2e-4  # Reduced learning rate for stability
    }
    
    # Initialize model and physics
    model = PINN_LMD(input_dim=4, hidden_dim=128, output_dim=1, params=params)
    physics = LMDPhysics(params)
    
    # Create trainer
    trainer = PINNTrainer(model, physics, params)
    
    # Train the model - reduced epochs for faster initial testing
    history = trainer.train(
        epochs=5000,  # Reduced from 10000 to 5000 for initial testing
        n_pde=3000,
        n_ic=1000,
        n_bc=600,     # Reduced for better handling (divisible by 6)
        weight_update_freq=500  # More frequent updates
    )
    
    # Plot the loss history
    plot_loss(history)
    
    # Plot temperature fields at different times and heights
    times = [0.5, 1.0, 2.0, 3.0]
    for t in times:
        plot_temperature_field(model, params, t, z_value=0.0)
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'history': history,
        'params': params
    }, 'pinn_lmd_model.pt')
    
    print("Training and visualization completed!")

if __name__ == "__main__":
    main()
