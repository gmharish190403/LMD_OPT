import torch

class Physics:
    def __init__(self, params, scale_x, scale_y, scale_z, scale_t, scale_u):
        """
        Initialize the physics module with process parameters and scaling factors
        """
        self.params = params
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.scale_t = scale_t
        self.scale_u = scale_u
        self.print_freq = 100
        
        # Reference value for PDE normalization
        self.Q_max = 2.15385e12

    def scale_inputs(self, X):
        """
        Scale the input coordinates correctly
        """
        # Properly scale spatial coordinates
        x = X[:, 0:1] * self.scale_x
        y = X[:, 1:2] * self.scale_y
        z = X[:, 2:3] * self.scale_z
        
        # Properly scale time - this is crucial for correct time-dependent behavior
        # The time values should be in range [0, 1] after scaling
        t = X[:, 3:4] * self.scale_t
        
        return torch.cat([x, y, z, t], dim=1)

    def scale_back_temperature(self, u_scaled):
        """
        Scale back temperature from normalized value to physical value
        """
        return u_scaled / self.scale_u

    def specific_heat(self, T, epoch=None):
        """
        Temperature-dependent specific heat capacity
        """
        T = self.scale_back_temperature(T)
        
        # Based on the paper value of 500 J/(kg·K)
        Cp = torch.full_like(T, 500.0)
        
        if epoch is not None and (epoch + 1) % self.print_freq == 0:
            print(f"Epoch {epoch+1} | T min in specific_heat: {T.min().item():.6f}, T max: {T.max().item():.6f}")
            print(f"Epoch {epoch+1} | Cp min: {Cp.min().item():.6f}, Cp max: {Cp.max().item():.6f}")
        
        return Cp

    def thermal_conductivity(self, T, epoch=None):
        """
        Temperature-dependent thermal conductivity
        """
        T = self.scale_back_temperature(T)
        
        # Following the relationship from the paper
        # For T < 1773.15K: k(T) = 2.0×10^-5·T² - 0.0444·T + 49.94
        # For T ≥ 1773.15K: k(T) = 1.04×10^-4·T² - 0.3426·T + 314.2
        cond1 = 2e-5 * T**2 - 0.0444 * T + 49.94
        cond2 = 1.04e-4 * T**2 - 0.3426 * T + 314.2
        
        k = torch.where(T < 1773.15, cond1, cond2)
        
        # Clamp to reasonable physical values
        k = torch.clamp(k, min=20, max=300)
        
        if epoch is not None and (epoch + 1) % self.print_freq == 0:
            print(f"Epoch {epoch+1} | k min: {k.min().item():.6f}, k max: {k.max().item():.6f}")
        
        return k

    def laser_heat_source(self, X, epoch=None):
        """
        Calculate the laser heat source (Gaussian distribution)
        """
        x = X[:, 0:1]
        y = X[:, 1:2]
        z = X[:, 2:3]
        t = X[:, 3:4]
        
        # FIXED: Calculate the actual time in seconds for laser position calculation
        # Unscale the time value to get actual seconds
        t_seconds = t / self.scale_t
        
        # Calculate laser position correctly
        # Laser is active only during deposition time (t1)
        is_active = t_seconds <= self.params['t1']
        
        # Calculate the distance from the moving laser center
        # The laser position along x-axis is v*t
        laser_pos_x = self.params['v'] * t_seconds
        
        # Compute squared distance from laser center
        r_squared = (((x / self.scale_x - laser_pos_x) / self.params['Ra'])**2 + 
                    ((y / self.scale_y - (self.params['Ly']/2)) / self.params['Rb'])**2 + 
                    ((z / self.scale_z) / self.params['Rc'])**2)
        
        # Gaussian volumetric heat source formula
        Q_coefficient = (6 * torch.sqrt(torch.tensor(3.0)) * self.params['eta'] * self.params['P']) / \
                       (torch.pi * torch.sqrt(torch.tensor(torch.pi)) * self.params['Ra'] * self.params['Rb'] * self.params['Rc'])
        
        Q = Q_coefficient * torch.exp(-3 * r_squared)
        
        # Set Q to zero when the laser is not active
        Q = torch.where(is_active, Q, torch.zeros_like(Q))
        
        if epoch is not None and (epoch + 1) % self.print_freq == 0:
            print(f"Epoch {epoch+1} | Using P: {self.params['P']}, eta: {self.params['eta']}, Ra: {self.params['Ra']}, Rb: {self.params['Rb']}, Rc: {self.params['Rc']}")
            print(f"Epoch {epoch+1} | Q min: {Q.min().item():.6e}, Q max: {Q.max().item():.6e}")
        
        return Q

    def pde_residual(self, X, u, epoch=None):
        """
        Calculate the PDE residual (heat equation)
        """
        # Ensure X requires gradient
        if not X.requires_grad:
            X.requires_grad_(True)
            
        # Calculate derivatives using automatic differentiation
        u_t = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 3:4]
        u_x = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 1:2]
        u_z = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 2:3]
        
        u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1:2]
        u_zz = torch.autograd.grad(u_z, X, grad_outputs=torch.ones_like(u_z), create_graph=True, retain_graph=True)[0][:, 2:3]
        
        # Convert scaled outputs to physical values
        T = self.scale_back_temperature(u)
        Cp = self.specific_heat(u, epoch)
        k = self.thermal_conductivity(u, epoch)
        Q = self.laser_heat_source(X, epoch)
        
        # Convert derivatives from scaled to physical space
        u_t_physical = u_t / (self.scale_t * self.scale_u)
        u_xx_physical = u_xx / (self.scale_x**2 * self.scale_u)
        u_yy_physical = u_yy / (self.scale_y**2 * self.scale_u)
        u_zz_physical = u_zz / (self.scale_z**2 * self.scale_u)
        
        # Calculate laplacian
        laplacian = u_xx_physical + u_yy_physical + u_zz_physical
        
        # Normalize PDE terms
        rho_Cp_u_t = (self.params['rho'] * Cp * u_t_physical) / self.Q_max
        k_laplacian = (k * laplacian) / self.Q_max
        Q_normalized = Q / self.Q_max
        
        # Heat equation residual (normalized): ρCp∂T/∂t - ∇·(k∇T) - Q = 0
        residual = rho_Cp_u_t - k_laplacian - Q_normalized
        
        if epoch is not None and (epoch + 1) % self.print_freq == 0:
            print(f"Epoch {epoch+1} | u min: {u.min().item():.6f}, u max: {u.max().item():.6f}")
            print(f"Epoch {epoch+1} | u_t min: {u_t_physical.min().item():.6f}, u_t max: {u_t_physical.max().item():.6f}")
            print(f"Epoch {epoch+1} | Laplacian min: {laplacian.min().item():.6f}, max: {laplacian.max().item():.6f}")
            print(f"Epoch {epoch+1} | Normalized residual min: {residual.min().item():.6f}, max: {residual.max().item():.6f}")
        
        return residual

    def ic_residual(self, u):
        """
        Calculate the initial condition residual
        """
        # Initial condition: T(x,y,z,0) = T0
        return u - self.params['T0'] * self.scale_u

    def bc_residual(self, X, u):
        """
        Calculate the boundary condition residual
        """
        # Ensure X requires gradient
        if not X.requires_grad:
            X.requires_grad_(True)
            
        # Calculate temperature gradients at boundaries
        u_x = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 1:2]
        u_z = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 2:3]
        
        # Convert to physical values
        T = self.scale_back_temperature(u)
        k = self.thermal_conductivity(u)
        
        # Convert gradients from scaled to physical space
        u_x_physical = u_x / (self.scale_x * self.scale_u)
        u_y_physical = u_y / (self.scale_y * self.scale_u)
        u_z_physical = u_z / (self.scale_z * self.scale_u)
        
        # Extract coordinates
        x = X[:, 0:1] / self.scale_x
        y = X[:, 1:2] / self.scale_y
        z = X[:, 2:3] / self.scale_z
        
        # Convection and radiation heat fluxes
        q_c = self.params['h'] * (T - self.params['T0'])
        q_r = self.params['sigma'] * self.params['epsilon'] * (T**4 - self.params['T0']**4)
        
        # Initialize flux residual
        flux = torch.zeros_like(u)
        
        # Define tolerance for boundary detection
        tol = 1e-6
        
        # Apply different boundary conditions
        # x = 0 (insulated)
        flux = torch.where(torch.abs(x) < tol, k * u_x_physical, flux)
        
        # x = Lx (convection + radiation)
        flux = torch.where(torch.abs(x - self.params['Lx']) < tol, 
                          -k * u_x_physical - (q_c + q_r), flux)
        
        # y = 0 (insulated)
        flux = torch.where(torch.abs(y) < tol, k * u_y_physical, flux)
        
        # y = Ly (insulated)
        flux = torch.where(torch.abs(y - self.params['Ly']) < tol, 
                          k * u_y_physical, flux)
        
        # z = 0 (insulated)
        flux = torch.where(torch.abs(z) < tol, k * u_z_physical, flux)
        
        # z = Lz (convection + radiation)
        flux = torch.where(torch.abs(z - self.params['Lz']) < tol, 
                          -k * u_z_physical - (q_c + q_r), flux)
        
        return flux

    def temperature_penalty(self, u):
        """
        Penalty to constrain temperature within physical bounds
        """
        T = self.scale_back_temperature(u)
        
        # Penalty for temperatures outside physical range
        lower_penalty = torch.relu(self.params['T0'] - T)
        upper_penalty = torch.relu(T - 5000.0)  # Unrealistically high temp
        
        return lower_penalty + upper_penalty
