import torch
import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train PINN for LMD process simulation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'plot'],
                       help='Mode to run (train, evaluate, plot)')
    parser.add_argument('--epochs', type=int, default=24000,  # Reduced epochs for better stability
                       help='Number of training epochs')
    parser.add_argument('--adam_epochs', type=int, default=10000,  # Paper value
                       help='Number of epochs to use Adam optimizer')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to load pretrained model (for evaluate/plot modes)')
    parser.add_argument('--plot_times', type=str, default='0.5,1.0,1.5,2.0',
                       help='Comma-separated list of time points for plotting')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--laser_power', type=float, default=2000.0,
                       help='Laser power in watts (2000W in paper)')
    
    args = parser.parse_args()
    
    # Import modules here to avoid potential circular imports
    from model import PINN
    from physics import Physics
    from utils import sample_points, get_monitor_points
    from train import train_pinn, predict_temperature
    from plot import (plot_loss_history, plot_temperature_field, 
                     plot_thermal_history, plot_melt_pool_evolution)
    
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
    
    # Problem parameters based on the paper
    params = {
        'rho': 7780.0,                # Density (kg/m^3)
        'eta': 0.75,                  # Laser absorption coefficient
        'P': args.laser_power,        # Laser power (W)
        'v': 8.0e-3,                  # Scanning speed (m/s)
        'Ra': 3.0e-3,                 # Laser spot radius in x (mm)
        'Rb': 3.0e-3,                 # Laser spot radius in y (mm)
        'Rc': 1.0e-3,                 # Laser spot radius in z (mm)
        'h': 20.0,                    # Convection coefficient (W/(m^2·K))
        'epsilon': 0.85,              # Emissivity
        'sigma': 5.67e-8,             # Stefan-Boltzmann constant (W/(m^2·K^4))
        'T0': 293.15,                 # Initial/ambient temperature (K)
        'Tm': 1730.0,                 # Melting temperature (K)
        'Ts': 1690.0,                 # Solidus temperature (K)
        'Lx': 0.04,                   # Domain length in x (m)
        'Ly': 0.02,                   # Domain width in y (m)
        'Lz': 0.005,                  # Domain height in z (m)
        't1': 2.0,                    # Deposition time (s)
        't2': 100.0                   # Cooling time (s)
    }
    
    # Scaling factors
    scale_x = 1.0 / params['Lx']
    scale_y = 1.0 / params['Ly']
    scale_z = 1.0 / params['Lz']
    scale_t = 1.0 / (params['t1'] + params['t2'])
    scale_u = 1.0 / 3000.0  # Max temperature expected
    
    # Initialize physics module
    physics = Physics(params, scale_x, scale_y, scale_z, scale_t, scale_u)
    
    # Loss weights - as specified in the paper
    weights = {
        'pde': 5.0,        # As in paper
        'ic': 1.0,         # As in paper
        'bc': 10.0,        # As in paper
        'temp': 0.1        # Small weight for temperature penalty
    }
    
    # Initialize or load model
    # In main.py, when creating the model
    model = PINN(num_neurons=60, num_blocks=2, input_dim=4, output_dim=1).to(device)
    
    # If in evaluate or plot mode, load pretrained model
    if args.mode in ['evaluate', 'plot'] and args.model_path:
        print(f"Loading model from {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"Model loaded with loss: {checkpoint.get('loss', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    
    # Run in the specified mode
    if args.mode == 'train':
        # Sample counts
        N_pde = 20000
        N_ic = 30000
        N_bc = 10000
        
        print(f"Generating training points: {N_pde} PDE, {N_ic} IC, {N_bc} BC")
        X_pde, X_ic, X_bc = sample_points(N_pde, N_ic, N_bc, params, device, non_uniform=True)
        
        # Ensure all inputs have requires_grad=True
        X_pde = X_pde.detach().requires_grad_(True)
        X_ic = X_ic.detach().requires_grad_(True) 
        X_bc = X_bc.detach().requires_grad_(True)
        
        # Train model
        print(f"Starting training for {args.epochs} epochs ({args.adam_epochs} Adam + {args.epochs - args.adam_epochs} L-BFGS)...")
        start_time = time.time()
        loss_history = train_pinn(model, physics, X_pde, X_ic, X_bc, args.epochs, args.adam_epochs,
                                 weights, device, save_dir=args.save_dir)
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f} seconds")
        
        # Plot loss history
        try:
            plot_loss_history(loss_history, args.save_dir)
        except Exception as e:
            print(f"Error plotting loss history: {e}")
            # Fallback simple plot
            plt.figure(figsize=(10, 6))
            plt.semilogy(range(len(loss_history['total'])), loss_history['total'], label='Total')
            plt.semilogy(range(len(loss_history['pde'])), loss_history['pde'], label='PDE')
            plt.semilogy(range(len(loss_history['ic'])), loss_history['ic'], label='IC')
            plt.semilogy(range(len(loss_history['bc'])), loss_history['bc'], label='BC')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, 'loss_history_simple.png'))
            plt.close()
        
        # Generate some initial evaluation plots
        try:
            plot_temperature_field(model, physics, 1.0, args.save_dir, plane='xy')
            plot_temperature_field(model, physics, 2.0, args.save_dir, plane='xy')
        except Exception as e:
            print(f"Error generating initial plots: {e}")
        
    elif args.mode == 'evaluate':
        if not args.model_path:
            print("Error: --model_path must be provided for evaluation mode")
            return
        
        # Parse plot times
        plot_times = [float(t) for t in args.plot_times.split(',')]
        
        # Monitor points
        monitor_points = get_monitor_points(params, device)
        point_labels = [
            f"Point 1 (10mm, {params['Ly']/2*1000:.1f}mm, 0mm)", 
            f"Point 2 (20mm, {params['Ly']/2*1000:.1f}mm, 0mm)", 
            f"Point 3 (30mm, {params['Ly']/2*1000:.1f}mm, 0mm)"
        ]
        
        # Generate evaluation time arrays
        # Deposition stage
        t_dep = torch.linspace(0, params['t1'], 100, device=device)
        # Cooling stage
        t_cool = torch.linspace(params['t1'], params['t1'] + 10.0, 100, device=device)  # Reduce cooling time for faster evaluation
        # Combined
        t_full = torch.cat([t_dep, t_cool[1:]])
        
        try:
            print("Evaluating thermal history at monitoring points...")
            temperature_histories = predict_temperature(model, physics, monitor_points, t_full, device)
            
            # Plot thermal history
            print("Plotting thermal history...")
            plot_thermal_history(model, physics, monitor_points, t_full, point_labels, args.save_dir)
            
            # Plotting temperature fields at specified times
            print(f"Plotting temperature fields at times: {plot_times}")
            for t in plot_times:
                plot_temperature_field(model, physics, t, args.save_dir, plane='xy')
                plot_temperature_field(model, physics, t, args.save_dir, plane='xz')
                plot_temperature_field(model, physics, t, args.save_dir, plane='yz')
            
            # Plot melt pool evolution (during deposition)
            print("Plotting melt pool evolution...")
            t_mp = torch.linspace(0.1, params['t1'], 10, device=device)
            plot_melt_pool_evolution(model, physics, t_mp, args.save_dir)
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
    elif args.mode == 'plot':
        # Generate visualization of key results from saved model
        print("Generating plots from saved model...")
        
        # Parse plot times
        plot_times = [float(t) for t in args.plot_times.split(',')]
        
        for t in plot_times:
            try:
                plot_temperature_field(model, physics, t, args.save_dir, plane='xy')
                plot_temperature_field(model, physics, t, args.save_dir, plane='xz')
            except Exception as e:
                print(f"Error during plotting time {t}: {e}")
    
    print(f"All results saved to {args.save_dir}")

if __name__ == "__main__":
    main()
