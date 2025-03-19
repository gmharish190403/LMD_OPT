import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import torch

def plot_loss_history(loss_history, save_path='results'):
    """
    Plot the loss curves during training
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
    
    # Ensure all arrays have the same length
    min_len = min(len(loss_history[key]) for key in loss_history if len(loss_history[key]) > 0)
    for key in loss_history:
        if len(loss_history[key]) > min_len:
            loss_history[key] = loss_history[key][:min_len]
        
    plt.figure(figsize=(12, 8))
    epochs = range(len(loss_history['total']))
    
    # Plot all loss components
    plt.semilogy(epochs, loss_history['total'], label='Total Loss', linewidth=2)
    plt.semilogy(epochs, loss_history['pde'], label='PDE Loss', linewidth=2)
    plt.semilogy(epochs, loss_history['ic'], label='IC Loss', linewidth=2)
    plt.semilogy(epochs, loss_history['bc'], label='BC Loss', linewidth=2)
    
    if 'temp' in loss_history and len(loss_history['temp']) > 0:
        plt.semilogy(epochs, loss_history['temp'], label='Temperature Constraint Loss', linewidth=2)
    
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

def plot_temperature_field(model, physics, time, save_path='results', plane='xy'):
    """
    Generate 2D contour plot of temperature field at a specific time
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Number of points in each dimension
    n1, n2 = 100, 50
    
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
        # Middle width (y = Ly/2)
        coord1 = torch.linspace(0, physics.params['Lx'], n1, device=device)
        coord2 = torch.linspace(0, physics.params['Lz'], n2, device=device)
        X1, X3 = torch.meshgrid(coord1, coord2, indexing='ij')
        X2 = torch.ones_like(X1) * (physics.params['Ly'] / 2)
        title = f'Temperature Field at t = {time:.2f}s (Middle Width, y = {physics.params["Ly"]/2*1000:.1f}mm)'
        xlabel, ylabel = 'X (mm)', 'Z (mm)'
    elif plane == 'yz':
        # Fixed x position
        coord1 = torch.linspace(0, physics.params['Ly'], n1, device=device)
        coord2 = torch.linspace(0, physics.params['Lz'], n2, device=device)
        X2, X3 = torch.meshgrid(coord1, coord2, indexing='ij')
        # Choose x position based on laser position at the given time
        if time <= physics.params['t1']:
            X1 = torch.ones_like(X2) * (physics.params['v'] * time)
            title = f'Temperature Field at t = {time:.2f}s (At Laser Position, x = {(physics.params["v"] * time)*1000:.1f}mm)'
        else:
            X1 = torch.ones_like(X2) * (physics.params['v'] * physics.params['t1'])
            title = f'Temperature Field at t = {time:.2f}s (Last Laser Position, x = {(physics.params["v"] * physics.params["t1"])*1000:.1f}mm)'
        xlabel, ylabel = 'Y (mm)', 'Z (mm)'
    
    # Reshape for batch processing
    coord1_flat = X1.flatten()
    coord2_flat = X2.flatten()
    coord3_flat = X3.flatten()
    
    # Create time tensor
    time_tensor = torch.ones_like(coord1_flat) * time
    
    # Stack coordinates (x, y, z, t)
    coords = torch.stack([coord1_flat, coord2_flat, coord3_flat, time_tensor], dim=1)
    
    # Scale inputs
    coords_scaled = physics.scale_inputs(coords)
    
    # Forward pass
    with torch.no_grad():
        u_scaled = model(coords_scaled)
        temperature = physics.scale_back_temperature(u_scaled)
    
    # Reshape to grid
    temperature_grid = temperature.reshape(X1.shape).cpu().numpy()
    
    # Convert coordinates to mm for plotting
    X1_mm = X1.cpu().numpy() * 1000
    X2_mm = X2.cpu().numpy() * 1000
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot temperature contours
    contour = plt.contourf(X1_mm, X2_mm, temperature_grid, 
                          levels=50, cmap='hot')
    
    # Add colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label('Temperature (K)', fontsize=12)
    
    # Add laser position marker if during deposition
    if time <= physics.params['t1'] and plane == 'xy':
        laser_pos_x = physics.params['v'] * time * 1000  # mm
        laser_pos_y = physics.params['Ly'] / 2 * 1000  # mm
        plt.plot(laser_pos_x, laser_pos_y, 'o', 
                markersize=10, color='cyan', label='Laser Position')
        plt.legend(fontsize=12)
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(False)
    
    # Add melting point contour
    if 'Tm' in physics.params:
        plt.contour(X1_mm, X2_mm, temperature_grid, 
                   levels=[physics.params['Tm']], 
                   colors='cyan', linewidths=2, 
                   linestyles='dashed')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'temperature_field_{plane}_t{time:.2f}.png'), dpi=300)
    plt.close()
    
    return temperature_grid

def plot_thermal_history(model, physics, monitor_points, times, point_labels=None, save_path='results'):
    """
    Plot thermal history at specific monitoring points
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Convert times to torch tensor if needed
    if not isinstance(times, torch.Tensor):
        times = torch.tensor(times, dtype=torch.float32, device=device)
    
    # If no labels provided, create generic ones
    if point_labels is None:
        point_labels = [f"Point {i+1}" for i in range(monitor_points.shape[0])]
    
    # Prepare figure
    plt.figure(figsize=(12, 8))
    
    # Use a colormap for different points
    colors = plt.cm.viridis(np.linspace(0, 1, monitor_points.shape[0]))
    
    # Track temperature histories
    temperature_histories = []
    
    with torch.no_grad():
        for i, point in enumerate(monitor_points):
            temperatures = []
            
            for t in times:
                # Create input with current point and time
                t_tensor = torch.tensor([t], device=device)
                X = torch.cat([point.unsqueeze(0), t_tensor.unsqueeze(0)], dim=1)
                
                # Scale inputs
                X_scaled = physics.scale_inputs(X)
                
                # Forward pass
                u_scaled = model(X_scaled)
                
                # Convert to physical temperature
                temperature = physics.scale_back_temperature(u_scaled)
                temperatures.append(temperature.item())
            
            temperature_histories.append(temperatures)
            
            # Plot
            plt.plot(times.cpu().numpy(), temperatures, label=point_labels[i], 
                     color=colors[i], linewidth=2)
    
    # Add vertical line at t1 (end of deposition)
    if 't1' in physics.params:
        plt.axvline(x=physics.params['t1'], color='k', linestyle='--', 
                    alpha=0.5, label=f"End of Deposition (t={physics.params['t1']}s)")
    
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Temperature (K)', fontsize=14)
    plt.title('Thermal History at Monitoring Points', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add melting point line if applicable
    if 'Tm' in physics.params:
        plt.axhline(y=physics.params['Tm'], color='r', linestyle='--', 
                    alpha=0.5, label=f"Melting Point ({physics.params['Tm']}K)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'thermal_history.png'), dpi=300)
    plt.close()
    
    return np.array(temperature_histories)

def plot_melt_pool_evolution(model, physics, times, save_path='results'):
    """
    Plot the evolution of melt pool size over time
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Track melt pool dimensions
    melt_pool_lengths = []
    melt_pool_widths = []
    melt_pool_depths = []
    
    # Melting temperature
    Tm = physics.params['Tm']
    
    for t in times:
        # Only calculate during deposition
        if t > physics.params['t1']:
            melt_pool_lengths.append(0)
            melt_pool_widths.append(0)
            melt_pool_depths.append(0)
            continue
        
        # Calculate melt pool dimensions at this time
        
        # 1. Length (x-direction)
        nx = 200
        x = torch.linspace(0, physics.params['Lx'], nx, device=device)
        y = torch.ones_like(x) * (physics.params['Ly'] / 2)
        z = torch.zeros_like(x)
        t_tensor = torch.ones_like(x) * t
        
        # Stack coordinates
        coords = torch.stack([x, y, z, t_tensor], dim=1)
        
        # Scale inputs
        coords_scaled = physics.scale_inputs(coords)
        
        # Get temperatures along this line
        with torch.no_grad():
            u_scaled = model(coords_scaled)
            temps_x = physics.scale_back_temperature(u_scaled).cpu().numpy()
        
        # Find melt pool length (where T > Tm)
        melt_mask = temps_x > Tm
        if np.any(melt_mask):
            x_mm = x.cpu().numpy() * 1000  # convert to mm
            melt_indices = np.where(melt_mask)[0]
            mp_length = x_mm[melt_indices.max()] - x_mm[melt_indices.min()]
        else:
            mp_length = 0
        
        melt_pool_lengths.append(mp_length)
        
        # 2. Width (y-direction)
        ny = 100
        y = torch.linspace(0, physics.params['Ly'], ny, device=device)
        x = torch.ones_like(y) * (physics.params['v'] * t)  # Laser position
        z = torch.zeros_like(y)
        t_tensor = torch.ones_like(y) * t
        
        # Stack coordinates
        coords = torch.stack([x, y, z, t_tensor], dim=1)
        
        # Scale inputs
        coords_scaled = physics.scale_inputs(coords)
        
        # Get temperatures along this line
        with torch.no_grad():
            u_scaled = model(coords_scaled)
            temps_y = physics.scale_back_temperature(u_scaled).cpu().numpy()
        
        # Find melt pool width (where T > Tm)
        melt_mask = temps_y > Tm
        if np.any(melt_mask):
            y_mm = y.cpu().numpy() * 1000  # convert to mm
            melt_indices = np.where(melt_mask)[0]
            mp_width = y_mm[melt_indices.max()] - y_mm[melt_indices.min()]
        else:
            mp_width = 0
        
        melt_pool_widths.append(mp_width)
        
        # 3. Depth (z-direction)
        nz = 50
        z = torch.linspace(0, physics.params['Lz'], nz, device=device)
        x = torch.ones_like(z) * (physics.params['v'] * t)  # Laser position
        y = torch.ones_like(z) * (physics.params['Ly'] / 2)
        t_tensor = torch.ones_like(z) * t
        
        # Stack coordinates
        coords = torch.stack([x, y, z, t_tensor], dim=1)
        
        # Scale inputs
        coords_scaled = physics.scale_inputs(coords)
        
        # Get temperatures along this line
        with torch.no_grad():
            u_scaled = model(coords_scaled)
            temps_z = physics.scale_back_temperature(u_scaled).cpu().numpy()
        
        # Find melt pool depth (where T > Tm)
        melt_mask = temps_z > Tm
        if np.any(melt_mask):
            z_mm = z.cpu().numpy() * 1000  # convert to mm
            melt_indices = np.where(melt_mask)[0]
            mp_depth = z_mm[melt_indices.max()]
        else:
            mp_depth = 0
        
        melt_pool_depths.append(mp_depth)
    
    # Convert to numpy arrays
    times_np = np.array(times)
    melt_pool_lengths = np.array(melt_pool_lengths)
    melt_pool_widths = np.array(melt_pool_widths)
    melt_pool_depths = np.array(melt_pool_depths)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    plt.plot(times_np, melt_pool_lengths, 'ro-', label='Length', linewidth=2)
    plt.plot(times_np, melt_pool_widths, 'go-', label='Width', linewidth=2)
    plt.plot(times_np, melt_pool_depths, 'bo-', label='Depth', linewidth=2)
    
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Melt Pool Dimension (mm)', fontsize=14)
    plt.title('Melt Pool Evolution During Deposition', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add vertical line at end of deposition
    plt.axvline(x=physics.params['t1'], color='k', linestyle='--', 
                alpha=0.5, label=f"End of Deposition (t={physics.params['t1']}s)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'melt_pool_evolution.png'), dpi=300)
    plt.close()
    
    return times_np, melt_pool_lengths, melt_pool_widths, melt_pool_depths
