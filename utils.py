import torch
import numpy as np

def scale_inputs(X, min_val, max_val):
    """
    Scale input values to the range [0, 1]
    """
    return (X - min_val) / (max_val - min_val)

def scale_back_temperature(u_scaled, scale_u):
    """
    Scale back temperature from normalized value to physical value
    """
    return u_scaled / scale_u

def sample_points(N_pde, N_ic, N_bc, domain, device, non_uniform=True):
    """
    Generate sampling points for PDE, initial and boundary conditions
    """
    # Implementation of non-uniform sampling as mentioned in the paper
    if non_uniform:
        print("Using non-uniform sampling strategy")
        
        # PDE points - use more points near the laser path and melt pool
        N_laser_path = N_pde // 2
        N_uniform = N_pde - N_laser_path
        
        # Uniform points throughout the domain (in physical units)
        x_uniform = torch.rand(N_uniform, 1) * domain['Lx']
        y_uniform = torch.rand(N_uniform, 1) * domain['Ly']
        z_uniform = torch.rand(N_uniform, 1) * domain['Lz']
        
        # Generate time points in physical seconds from 0 to t1+t2
        t_uniform = torch.rand(N_uniform, 1) * (domain['t1'] + domain['t2'])
        
        # Non-uniform points concentrated around laser path
        laser_path_end = domain['v'] * domain['t1']
        
        # Generate points clustered around laser path with more density at early times
        # Time points during deposition phase (0 to t1 seconds)
        t_laser = torch.rand(N_laser_path, 1).pow(0.5) * domain['t1']
        
        # Positions along laser path (0 to v*t1)
        x_laser = torch.rand(N_laser_path, 1) * laser_path_end
        
        # Add gaussian noise around the path to focus on the melt pool region
        laser_width = 0.005  # 5mm spread around path
        y_laser = domain['Ly']/2 + torch.randn(N_laser_path, 1) * laser_width
        y_laser = torch.clamp(y_laser, 0, domain['Ly']) 
        
        # Focus more points near the surface (z=0)
        z_laser = torch.rand(N_laser_path, 1).pow(2) * domain['Lz']
        
        # Combine uniform and non-uniform points
        x_pde = torch.cat([x_uniform, x_laser], dim=0)
        y_pde = torch.cat([y_uniform, y_laser], dim=0)
        z_pde = torch.cat([z_uniform, z_laser], dim=0)
        t_pde = torch.cat([t_uniform, t_laser], dim=0)
    else:
        # Uniform sampling (in physical units)
        x_pde = torch.rand(N_pde, 1) * domain['Lx']
        y_pde = torch.rand(N_pde, 1) * domain['Ly']
        z_pde = torch.rand(N_pde, 1) * domain['Lz']
        t_pde = torch.rand(N_pde, 1) * (domain['t1'] + domain['t2'])
    
    # Stack and enable gradients
    X_pde = torch.cat([x_pde, y_pde, z_pde, t_pde], dim=1).to(device).requires_grad_(True)
    
    # Print time range for debugging
    print(f"PDE points time range: {t_pde.min().item():.4f} to {t_pde.max().item():.4f} seconds")
    
    # IC points (t = 0)
    if non_uniform:
        # Higher density near the surface and center
        x_ic = torch.rand(N_ic, 1) * domain['Lx']
        y_ic = torch.rand(N_ic, 1) * domain['Ly']
        z_ic = torch.rand(N_ic, 1).pow(2) * domain['Lz']
    else:
        x_ic = torch.rand(N_ic, 1) * domain['Lx']
        y_ic = torch.rand(N_ic, 1) * domain['Ly']
        z_ic = torch.rand(N_ic, 1) * domain['Lz']
    
    # Initial condition time is always t=0
    t_ic = torch.zeros(N_ic, 1)
    X_ic = torch.cat([x_ic, y_ic, z_ic, t_ic], dim=1).to(device).requires_grad_(True)
    
    # BC points on the 6 boundary surfaces
    N_bc_per_face = N_bc // 6
    
    # Lists to store boundary points for each face
    bc_points = []
    
    # Sample times non-uniformly (more points during deposition)
    if non_uniform:
        t_bc = torch.rand(N_bc_per_face, 1).pow(0.5) * (domain['t1'] + domain['t2'])
    else:
        t_bc = torch.rand(N_bc_per_face, 1) * (domain['t1'] + domain['t2'])
    
    # Print boundary time range for debugging
    print(f"Boundary points time range: {t_bc.min().item():.4f} to {t_bc.max().item():.4f} seconds")
    
    # 1. x = 0 face (left)
    x_face = torch.zeros(N_bc_per_face, 1)
    y_face = torch.rand(N_bc_per_face, 1) * domain['Ly']
    z_face = torch.rand(N_bc_per_face, 1) * domain['Lz']
    bc_points.append(torch.cat([x_face, y_face, z_face, t_bc.clone()], dim=1))
    
    # 2. x = Lx face (right)
    x_face = torch.ones(N_bc_per_face, 1) * domain['Lx']
    y_face = torch.rand(N_bc_per_face, 1) * domain['Ly']
    z_face = torch.rand(N_bc_per_face, 1) * domain['Lz']
    bc_points.append(torch.cat([x_face, y_face, z_face, t_bc.clone()], dim=1))
    
    # 3. y = 0 face (front)
    x_face = torch.rand(N_bc_per_face, 1) * domain['Lx']
    y_face = torch.zeros(N_bc_per_face, 1)
    z_face = torch.rand(N_bc_per_face, 1) * domain['Lz']
    bc_points.append(torch.cat([x_face, y_face, z_face, t_bc.clone()], dim=1))
    
    # 4. y = Ly face (back)
    x_face = torch.rand(N_bc_per_face, 1) * domain['Lx']
    y_face = torch.ones(N_bc_per_face, 1) * domain['Ly']
    z_face = torch.rand(N_bc_per_face, 1) * domain['Lz']
    bc_points.append(torch.cat([x_face, y_face, z_face, t_bc.clone()], dim=1))
    
    # 5. z = 0 face (bottom)
    x_face = torch.rand(N_bc_per_face, 1) * domain['Lx']
    y_face = torch.rand(N_bc_per_face, 1) * domain['Ly']
    z_face = torch.zeros(N_bc_per_face, 1)
    bc_points.append(torch.cat([x_face, y_face, z_face, t_bc.clone()], dim=1))
    
    # 6. z = Lz face (top)
    x_face = torch.rand(N_bc_per_face, 1) * domain['Lx']
    y_face = torch.rand(N_bc_per_face, 1) * domain['Ly']
    z_face = torch.ones(N_bc_per_face, 1) * domain['Lz']
    bc_points.append(torch.cat([x_face, y_face, z_face, t_bc.clone()], dim=1))
    
    # Concatenate all boundary points
    X_bc = torch.cat(bc_points, dim=0).to(device).requires_grad_(True)
    
    return X_pde, X_ic, X_bc

def get_monitor_points(domain, device):
    """
    Create monitor points for temperature history tracking
    """
    # Monitor points at the top surface
    points = [
        [0.010, domain['Ly']/2, 0.0],  # Point 1 (10mm, middle width, top)
        [0.020, domain['Ly']/2, 0.0],  # Point 2 (20mm, middle width, top)
        [0.030, domain['Ly']/2, 0.0]   # Point 3 (30mm, middle width, top)
    ]
    
    return torch.tensor(points, dtype=torch.float32, device=device)
