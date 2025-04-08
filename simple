import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# Adaptive Tanh Activation Function
class AdaptiveTanh(nn.Module):
    """Layer-wise Locally Adaptive Tanh Activation"""
    def __init__(self, n=10):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1) * n)

    def forward(self, x):
        return torch.tanh(self.a * x)

# Residual Block
class ResidualBlock(nn.Module):
    """Residual block with adaptive activation"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = AdaptiveTanh()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)

        # Xavier initialization
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        return out + identity  # Skip connection

# PINN Model
class PINN_LMD(nn.Module):
    """Physics-Informed Neural Network for LMD"""
    def __init__(self, input_dim=4, hidden_dim=50, output_dim=1):
        super().__init__()

        # Scaling and normalization buffers
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('k_x', torch.ones(1))
        self.register_buffer('k_t', torch.ones(1))
        self.register_buffer('k_u', torch.ones(1))

        # Network architecture
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.res_block = ResidualBlock(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_normal_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

    def set_scaling_factors(self, Lx, Ly, Lz, t1, t2, T_max):
        """Set scaling factors based on domain size and max temperature"""
        self.k_x.data = torch.tensor(1.0 / max(Lx, Ly, Lz), dtype=torch.float32)
        self.k_t.data = torch.tensor(1.0 / (t1 + t2), dtype=torch.float32)
        self.k_u.data = torch.tensor(1.0 / T_max, dtype=torch.float32)

    def set_input_stats(self, mean, std):
        """Set input normalization statistics"""
        self.input_mean.data.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.input_std.data.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x):
        # Normalize and scale inputs
        x_norm = (x - self.input_mean) / self.input_std
        x_scaled = torch.cat([x_norm[:, :3] * self.k_x, x_norm[:, 3:4] * self.k_t], dim=1)

        # Forward pass
        h = torch.tanh(self.input_layer(x_scaled))
        h = torch.tanh(self.hidden1(h))
        h = self.res_block(h)
        h = torch.tanh(h)
        u_scaled = torch.sigmoid(self.output_layer(h))
        return u_scaled / self.k_u  # Unscaled temperature

# Physics Module
class LMDPhysics:
    """Physics module for LMD-specific equations"""
    def __init__(self, params):
        self.params = params
        self.sigma_sb = 5.67e-8  # Stefan-Boltzmann constant
        self.Lx, self.Ly, self.Lz = params['Lx'], params['Ly'], params['Lz']
        self.t1, self.t2 = params['t1'], params['t2']
        self.Q_max = 5.0e11  # Max laser power density for normalization
        self.T_ref = 3000.0  # Reference temperature for scaling

    def thermal_conductivity(self, T):
        """Temperature-dependent thermal conductivity from the paper's Table
        Units: W/(m·K)
        """
        # For T < 1773.15 K
        k_low = 2.0e-5 * T**2 - 0.0444 * T + 49.94
        
        # For T ≥ 1773.15 K
        k_high = 1.04e-4 * T**2 - 0.3426 * T + 314.2
        
        # Combined piecewise function
        return torch.where(T < 1773.15, k_low, k_high)

    def specific_heat(self, T):
        """Temperature-dependent specific heat (Table 2)"""
        cp_low = 1.04e-4 * T**2 - 0.3426 * T + 314.2  # T < 1373.15 K
        cp_high = torch.full_like(T, 700.0)           # T >= 1373.15 K
        return torch.where(T < 1373.15, cp_low, cp_high)


    def laser_heat_source(self, x, y, z, t):
        """Gaussian laser heat source (Eq. 4)"""
        v, P = self.params['v'], self.params['P']
        eta, Ra, Rb, Rc = self.params['eta'], self.params['Ra'], self.params['Rb'], self.params['Rc']
        x0, y0, z0 = 0.0, self.Ly / 2, 0.0

        active = (t <= self.t1).float()
        r_sq = ((x - (v * t + x0)) / Ra)**2 + ((y - y0) / Rb)**2 + ((z - z0) / Rc)**2
        Q = (6 * np.sqrt(3) * eta * P / (np.pi * np.sqrt(np.pi) * Ra * Rb * Rc)) * torch.exp(-3 * r_sq) * active
        return Q / self.Q_max

    def compute_derivatives(self, model, x, y, z, t):
        """Compute derivatives using automatic differentiation"""
        coords = [x.clone().requires_grad_(True), y.clone().requires_grad_(True),
                  z.clone().requires_grad_(True), t.clone().requires_grad_(True)]
        inputs = torch.stack(coords, dim=1)
        T = model(inputs)

        grad_outputs = torch.ones_like(T)
        dT_dx, dT_dy, dT_dz, dT_dt = grad(T, coords, grad_outputs=grad_outputs, create_graph=True)

        d2T_dx2 = grad(dT_dx, coords[0], grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0]
        d2T_dy2 = grad(dT_dy, coords[1], grad_outputs=torch.ones_like(dT_dy), create_graph=True)[0]
        d2T_dz2 = grad(dT_dz, coords[2], grad_outputs=torch.ones_like(dT_dz), create_graph=True)[0]

        return T, dT_dt, dT_dx, dT_dy, dT_dz, d2T_dx2, d2T_dy2, d2T_dz2

    def pde_residual(self, model, x, y, z, t):
        """PDE residual (Eq. 1 and 5)"""
        T, dT_dt, dT_dx, dT_dy, dT_dz, d2T_dx2, d2T_dy2, d2T_dz2 = self.compute_derivatives(model, x, y, z, t)
        rho = self.params['rho']
        Cp = self.specific_heat(T)
        k = self.thermal_conductivity(T)
        Q = self.laser_heat_source(x, y, z, t)
        residual = rho * Cp * dT_dt - k * (d2T_dx2 + d2T_dy2 + d2T_dz2) - Q * self.Q_max
        return residual / self.Q_max

    def ic_residual(self, model, x, y, z, t):
        """Initial condition residual (Eq. 2 and 6)"""
        inputs = torch.stack([x, y, z, t], dim=1)
        T = model(inputs)
        return (T - self.params['T0']) / self.T_ref

    def bc_residual(self, model, x, y, z, t):
        """Boundary condition residual (Eq. 3 and 7)"""
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

        bc_residual = torch.where(is_x_min, dT_dx, bc_residual)
        bc_residual = torch.where(is_x_max, k * dT_dx + q_conv + q_rad, bc_residual)
        bc_residual = torch.where(is_y_min | is_y_max, dT_dy, bc_residual)
        bc_residual = torch.where(is_z_min, dT_dz, bc_residual)
        bc_residual = torch.where(is_z_max, k * dT_dz + q_conv + q_rad, bc_residual)
        return bc_residual / self.Q_max

# Training Point Generation
def generate_training_points(params, n_pde=3000, n_ic=4000, n_bc=2000, device='cpu', stage='deposition'):
    """Generate training points with focused sampling"""
    Lx, Ly, Lz = params['Lx'], params['Ly'], params['Lz']
    t1, t2 = params['t1'], params['t2']
    v = params['v']
    t_max = t1 if stage == 'deposition' else t2

    # PDE points
    n_laser = int(0.7 * n_pde)
    t_laser = torch.rand(n_laser, device=device).pow(2) * t_max
    x_laser = v * t_laser + 0.0005 * torch.randn(n_laser, device=device)
    y_laser = Ly / 2 + 0.001 * torch.randn(n_laser, device=device)
    z_laser = 0.0002 * torch.rand(n_laser, device=device).pow(3)

    n_surface = int(0.2 * n_pde)
    x_surface = torch.rand(n_surface, device=device) * Lx
    y_surface = torch.rand(n_surface, device=device) * Ly
    z_surface = 0.0001 * torch.rand(n_surface, device=device).pow(4)
    t_surface = torch.rand(n_surface, device=device) * t_max

    n_uniform = n_pde - n_laser - n_surface
    x_uniform = torch.rand(n_uniform, device=device) * Lx
    y_uniform = torch.rand(n_uniform, device=device) * Ly
    z_uniform = torch.rand(n_uniform, device=device) * Lz
    t_uniform = torch.rand(n_uniform, device=device) * t_max

    X_pde = torch.stack([torch.cat([x_laser, x_surface, x_uniform]),
                         torch.cat([y_laser, y_surface, y_uniform]),
                         torch.cat([z_laser, z_surface, z_uniform]),
                         torch.cat([t_laser, t_surface, t_uniform])], dim=1)

    # Initial condition points
    if stage == 'deposition':
        X_ic = torch.stack([torch.rand(n_ic, device=device) * Lx,
                            torch.rand(n_ic, device=device) * Ly,
                            torch.rand(n_ic, device=device) * Lz,
                            torch.zeros(n_ic, device=device)], dim=1)
    else:
        pass  # IC for cooling set externally

    # Boundary condition points
    n_per_face = n_bc // 6
    t_bc = torch.rand(n_per_face, device=device) * t_max
    faces = []
    for i in range(6):
        if i == 0:  # x=0
            x = torch.zeros(n_per_face, device=device)
            y = torch.rand(n_per_face, device=device) * Ly
            z = torch.rand(n_per_face, device=device) * Lz
        elif i == 1:  # x=Lx
            x = torch.ones(n_per_face, device=device) * Lx
            y = torch.rand(n_per_face, device=device) * Ly
            z = torch.rand(n_per_face, device=device) * Lz
        elif i == 2:  # y=0
            x = torch.rand(n_per_face, device=device) * Lx
            y = torch.zeros(n_per_face, device=device)
            z = torch.rand(n_per_face, device=device) * Lz
        elif i == 3:  # y=Ly
            x = torch.rand(n_per_face, device=device) * Lx
            y = torch.ones(n_per_face, device=device) * Ly
            z = torch.rand(n_per_face, device=device) * Lz
        elif i == 4:  # z=0
            x = torch.rand(n_per_face, device=device) * Lx
            y = torch.rand(n_per_face, device=device) * Ly
            z = torch.zeros(n_per_face, device=device)
        else:  # z=Lz
            x = torch.rand(n_per_face, device=device) * Lx
            y = torch.rand(n_per_face, device=device) * Ly
            z = torch.ones(n_per_face, device=device) * Lz
        faces.append(torch.stack([x, y, z, t_bc], dim=1))
    X_bc = torch.cat(faces)

    if stage == 'deposition':
        return X_pde, X_ic, X_bc
    return X_pde, X_bc

# Training Function
def train_pinn(model, physics, X_pde, X_ic, X_bc, device, epochs=3000, lbfgs_epochs=1600, stage='deposition'):
    """Two-phase training (Adam + L-BFGS)"""
    x_pde, y_pde, z_pde, t_pde = X_pde[:, 0], X_pde[:, 1], X_pde[:, 2], X_pde[:, 3]
    if stage == 'deposition':
        x_ic, y_ic, z_ic, t_ic = X_ic[:, 0], X_ic[:, 1], X_ic[:, 2], X_ic[:, 3]
    x_bc, y_bc, z_bc, t_bc = X_bc[:, 0], X_bc[:, 1], X_bc[:, 2], X_bc[:, 3]

    loss_weights = {'pde': 10.0, 'ic': 1.0, 'bc': 5.0}
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500)
    history = {'total': [], 'pde': [], 'ic': [], 'bc': []}

    # Adam optimization
    print(f"Training {stage} stage with Adam...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_pde = torch.mean(physics.pde_residual(model, x_pde, y_pde, z_pde, t_pde)**2)
        if stage == 'deposition':
            loss_ic = torch.mean(physics.ic_residual(model, x_ic, y_ic, z_ic, t_ic)**2)
        else:
            loss_ic = torch.mean((model(X_ic) - physics.T_ic)**2)
        loss_bc = torch.mean(physics.bc_residual(model, x_bc, y_bc, z_bc, t_bc)**2)
        total_loss = loss_weights['pde'] * loss_pde + loss_weights['ic'] * loss_ic + loss_weights['bc'] * loss_bc

        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        history['total'].append(total_loss.item())
        history['pde'].append(loss_pde.item())
        history['ic'].append(loss_ic.item())
        history['bc'].append(loss_bc.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: Loss={total_loss.item():.3e}")

    # L-BFGS optimization
    if lbfgs_epochs > 0:
        print(f"Training {stage} stage with L-BFGS...")
        lbfgs_optimizer = optim.LBFGS(model.parameters(), max_iter=lbfgs_epochs, tolerance_grad=1e-11,
                                      tolerance_change=1e-14, history_size=50, line_search_fn='strong_wolfe')
        lbfgs_iter = [0]

        def closure():
            lbfgs_optimizer.zero_grad()
            loss_pde = torch.mean(physics.pde_residual(model, x_pde, y_pde, z_pde, t_pde)**2)
            if stage == 'deposition':
                loss_ic = torch.mean(physics.ic_residual(model, x_ic, y_ic, z_ic, t_ic)**2)
            else:
                loss_ic = torch.mean((model(X_ic) - physics.T_ic)**2)
            loss_bc = torch.mean(physics.bc_residual(model, x_bc, y_bc, z_bc, t_bc)**2)
            total_loss = loss_weights['pde'] * loss_pde + loss_weights['ic'] * loss_ic + loss_weights['bc'] * loss_bc
            total_loss.backward()

            lbfgs_iter[0] += 1
            if lbfgs_iter[0] % 100 == 0:
                print(f"L-BFGS Iter {lbfgs_iter[0]}: Loss={total_loss.item():.3e}")
            return total_loss

        lbfgs_optimizer.step(closure)

    return history

# Plotting Function
def plot_temperature_field(model, physics, time, plane='xy', save_path=None):
    """Visualize temperature field with max temperature annotation"""
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device  # Get device (CPU/GPU)
    n_points = 200  # Number of points for grid

    # Define grid based on plane (xy or xz)
    if plane == 'xy':
        x = torch.linspace(0, physics.Lx, n_points, device=device)
        y = torch.linspace(0, physics.Ly, n_points, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        Z = torch.zeros_like(X)
        xlabel, ylabel = 'X (m)', 'Y (m)'
    else:  # xz plane
        x = torch.linspace(0, physics.Lx, n_points, device=device)
        z = torch.linspace(0, physics.Lz, n_points, device=device)
        X, Z = torch.meshgrid(x, z, indexing='ij')
        Y = torch.ones_like(X) * physics.Ly / 2
        xlabel, ylabel = 'X (m)', 'Z (m)'

    # Prepare input tensor with time
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), 
                               torch.ones_like(X.flatten()) * time], dim=1)

    # Predict temperature
    with torch.no_grad():
        T = model(grid_points).cpu().numpy().reshape(X.shape)

    # Calculate max temperature and its location
    T_max = np.max(T)
    idx_max = np.unravel_index(np.argmax(T), T.shape)
    x_max, y_max = X[idx_max].item(), Y[idx_max].item()

    # Create the plot
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), T, levels=50, cmap='inferno')
    plt.colorbar(contour, label='Temperature (K)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Temperature at t={time:.2f}s ({plane} plane)')
    
    # Add max temperature annotation
    plt.scatter(x_max, y_max, color='white', s=50, edgecolor='black', 
                label=f'Max T: {T_max:.2f} K at ({x_max:.3f}, {y_max:.3f})')
    plt.legend(loc='upper right')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# Main Execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {
        'rho': 7780.0, 'eta': 0.75, 'P': 2000.0, 'v': 8.0e-3,
        'Ra': 3.0e-3, 'Rb': 3.0e-3, 'Rc': 1.0e-3,
        'h': 20.0, 'epsilon': 0.85, 'T0': 293.15,
        'Tm': 1730.0, 'Lx': 0.04, 'Ly': 0.02, 'Lz': 0.005,
        't1': 2.0, 't2': 10.0  # Use 10s for testing, change to 100.0 for full run
    }

    # Deposition Stage
    model_dep = PINN_LMD().to(device)
    model_dep.set_scaling_factors(params['Lx'], params['Ly'], params['Lz'], params['t1'], 0.0, 3000.0)
    input_mean_dep = [params['Lx']/2, params['Ly']/2, params['Lz']/2, params['t1']/2]
    input_std_dep = [params['Lx'], params['Ly'], params['Lz'], params['t1']]
    model_dep.set_input_stats(input_mean_dep, input_std_dep)
    physics_dep = LMDPhysics(params)
    X_pde_dep, X_ic_dep, X_bc_dep = generate_training_points(params, device=device, stage='deposition')
    train_pinn(model_dep, physics_dep, X_pde_dep, X_ic_dep, X_bc_dep, device, stage='deposition')

    # Extract temperature at t=t1
    n_grid = 50
    x_grid = torch.linspace(0, params['Lx'], n_grid, device=device)
    y_grid = torch.linspace(0, params['Ly'], n_grid, device=device)
    z_grid = torch.linspace(0, params['Lz'], n_grid, device=device)
    X, Y, Z = torch.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), torch.ones_like(X.flatten()) * params['t1']], dim=1)
    with torch.no_grad():
        T_t1 = model_dep(grid_points)

    # Cooling Stage
    params_cool = params.copy()
    params_cool['t1'] = 0.0
    model_cool = PINN_LMD().to(device)
    model_cool.set_scaling_factors(params['Lx'], params['Ly'], params['Lz'], 0.0, params['t2'], 3000.0)
    input_mean_cool = [params['Lx']/2, params['Ly']/2, params['Lz']/2, params['t2']/2]
    input_std_cool = [params['Lx'], params['Ly'], params['Lz'], params['t2']]
    model_cool.set_input_stats(input_mean_cool, input_std_cool)
    physics_cool = LMDPhysics(params_cool)
    physics_cool.T_ic = T_t1
    X_pde_cool, X_bc_cool = generate_training_points(params_cool, device=device, stage='cooling')
    X_ic_cool = torch.cat([grid_points[:, :3], torch.zeros(grid_points.shape[0], 1, device=device)], dim=1)
    train_pinn(model_cool, physics_cool, X_pde_cool, X_ic_cool, X_bc_cool, device, stage='cooling')

    # Plot results
    for t in [0.5, 1.0, 2.0]:
        plot_temperature_field(model_dep, physics_dep, t, 'xy', f'temp_xy_dep_t{t:.1f}.png')
    for t in [2.5, 5.0, 10.0]:
        plot_temperature_field(model_cool, physics_cool, t - params['t1'], 'xy', f'temp_xy_cool_t{t:.1f}.png')

if __name__ == "__main__":
    main()
