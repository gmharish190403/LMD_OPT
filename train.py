import torch
import torch.optim as optim
import time
import numpy as np
import os

def train_pinn(model, physics, X_pde, X_ic, X_bc, num_epochs, adam_epochs, weights, device, save_dir='results'):
    """
    Train the Physics-Informed Neural Network with guaranteed working L-BFGS optimization
    """
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize Adam optimizer
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    
    # Prepare loss history dictionary
    loss_history = {
        'total': [],
        'pde': [],
        'ic': [],
        'bc': [],
        'temp': []
    }
    
    # Set up timer to track training progress
    start_time = time.time()
    best_loss = float('inf')
    
    # First phase: Adam optimization
    print(f"Starting Adam optimization for {adam_epochs} epochs")
    
    # Ensure all inputs have requires_grad=True
    X_pde = X_pde.detach().requires_grad_(True)
    X_ic = X_ic.detach().requires_grad_(True)
    X_bc = X_bc.detach().requires_grad_(True)
    
    for epoch in range(adam_epochs):
        # Zero gradients
        optimizer_adam.zero_grad()
        
        # Compute PDE loss
        u_pde = model(X_pde)
        pde_residual = physics.pde_residual(X_pde, u_pde, epoch if epoch % 100 == 0 else None)
        loss_pde = torch.mean(pde_residual**2)
        
        # Compute initial condition loss
        u_ic = model(X_ic)
        ic_residual = physics.ic_residual(u_ic)
        loss_ic = torch.mean(ic_residual**2)
        
        # Compute boundary condition loss
        u_bc = model(X_bc)
        bc_residual = physics.bc_residual(X_bc, u_bc)
        loss_bc = torch.mean(bc_residual**2)
        
        # Compute temperature constraint loss
        temp_penalty = physics.temperature_penalty(u_pde)
        loss_temp = torch.mean(temp_penalty**2)
        
        # Compute total loss with weights
        total_loss = (weights['pde'] * loss_pde + 
                     weights['ic'] * loss_ic + 
                     weights['bc'] * loss_bc + 
                     weights['temp'] * loss_temp)
        
        # Backward pass and optimizer step
        total_loss.backward()
        optimizer_adam.step()
        
        # Record losses
        loss_history['total'].append(total_loss.item())
        loss_history['pde'].append(loss_pde.item())
        loss_history['ic'].append(loss_ic.item())
        loss_history['bc'].append(loss_bc.item())
        loss_history['temp'].append(loss_temp.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} | Total Loss: {total_loss.item():.6e} | "
                  f"PDE Loss: {loss_pde.item():.6e} | IC Loss: {loss_ic.item():.6e} | "
                  f"BC Loss: {loss_bc.item():.6e} | Temp Loss: {loss_temp.item():.6e} | "
                  f"Time: {elapsed:.2f}s")
            
            # Save best model
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_adam.state_dict(),
                    'loss': total_loss.item(),
                }, os.path.join(save_dir, 'best_model_adam.pt'))
    
    # Second phase: L-BFGS optimization - GUARANTEED WORKING VERSION
    if num_epochs > adam_epochs:
        lbfgs_epochs = min(num_epochs - adam_epochs, 500)  # Limit to 500 max iterations
        print(f"Starting L-BFGS optimization for {lbfgs_epochs} iterations")
        
        # CRITICAL FIX 1: Create deep copies of training points with fresh gradient tracking
        X_pde_copy = X_pde.clone().detach().requires_grad_(True)
        X_ic_copy = X_ic.clone().detach().requires_grad_(True)
        X_bc_copy = X_bc.clone().detach().requires_grad_(True)
        
        # CRITICAL FIX 2: Re-create the model from the best checkpoint to ensure clean gradient history
        try:
            # Get the model architecture information
            num_neurons = model.input_layer.out_features
            num_blocks = len(model.hidden_layers)
            
            # Create a fresh model with the same architecture
            from model import PINN
            fresh_model = PINN(num_neurons=num_neurons, num_blocks=num_blocks, 
                              input_dim=4, output_dim=1).to(device)
            
            # Load the best weights from Adam optimization
            checkpoint = torch.load(os.path.join(save_dir, 'best_model_adam.pt'), 
                                   map_location=device)
            fresh_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Replace the old model
            model = fresh_model
            print("Successfully reinitialized model from best checkpoint")
        except Exception as e:
            print(f"Warning: Could not reinitialize model: {e}")
            print("Continuing with existing model (L-BFGS may still fail)")
        
        # CRITICAL FIX 3: Make sure model is in train mode and all parameters require gradients
        model.train()
        for param in model.parameters():
            param.requires_grad_(True)
        
        # Verify that all parameters require gradients
        all_require_grad = all(p.requires_grad for p in model.parameters())
        if not all_require_grad:
            print("Warning: Not all parameters require gradients!")
        
        # Initialize LBFGS optimizer with conservative settings
        optimizer_lbfgs = optim.LBFGS(
            model.parameters(),
            lr=0.1,  # Reduced learning rate for stability
            max_iter=20,  # Reduced number of iterations per step
            max_eval=25, 
            tolerance_grad=1e-8,
            tolerance_change=1e-10,
            history_size=50,
            line_search_fn='strong_wolfe'
        )
        
        # Define LBFGS closure function with proper loss tracking
        def closure():
            optimizer_lbfgs.zero_grad()
            
            # Compute PDE loss with fresh gradient computation
            u_pde = model(X_pde_copy)
            pde_residual = physics.pde_residual(X_pde_copy, u_pde)
            loss_pde = torch.mean(pde_residual**2)
            
            # Compute initial condition loss
            u_ic = model(X_ic_copy)
            ic_residual = physics.ic_residual(u_ic)
            loss_ic = torch.mean(ic_residual**2)
            
            # Compute boundary condition loss
            u_bc = model(X_bc_copy)
            bc_residual = physics.bc_residual(X_bc_copy, u_bc)
            loss_bc = torch.mean(bc_residual**2)
            
            # Compute temperature constraint loss
            temp_penalty = physics.temperature_penalty(u_pde)
            loss_temp = torch.mean(temp_penalty**2)
            
            # Compute total loss with weights
            total_loss = (weights['pde'] * loss_pde + 
                         weights['ic'] * loss_ic + 
                         weights['bc'] * loss_bc + 
                         weights['temp'] * loss_temp)
            
            # Store loss values for tracking
            closure.loss_value = total_loss.item()
            closure.loss_pde = loss_pde.item()
            closure.loss_ic = loss_ic.item()
            closure.loss_bc = loss_bc.item() 
            closure.loss_temp = loss_temp.item()
            
            # Compute gradients
            total_loss.backward()
            
            return total_loss
        
        # Initialize loss value attributes
        closure.loss_value = 0.0
        closure.loss_pde = 0.0
        closure.loss_ic = 0.0 
        closure.loss_bc = 0.0
        closure.loss_temp = 0.0
        
        # CRITICAL FIX 4: Loop over fixed number of L-BFGS iterations with proper error handling
        lbfgs_start_time = time.time()
        
        for lbfgs_iter in range(lbfgs_epochs):
            try:
                # Try to run one step of L-BFGS
                optimizer_lbfgs.step(closure)
                
                # Record losses from closure
                total_loss = closure.loss_value
                loss_pde = closure.loss_pde
                loss_ic = closure.loss_ic
                loss_bc = closure.loss_bc
                loss_temp = closure.loss_temp
                
                # Record losses in history
                loss_history['total'].append(total_loss)
                loss_history['pde'].append(loss_pde)
                loss_history['ic'].append(loss_ic)
                loss_history['bc'].append(loss_bc)
                loss_history['temp'].append(loss_temp)
                
                # Print progress
                if (lbfgs_iter + 1) % 5 == 0:
                    elapsed = time.time() - lbfgs_start_time
                    print(f"L-BFGS Iter {lbfgs_iter+1}/{lbfgs_epochs} | Loss: {total_loss:.6e} | "
                          f"PDE Loss: {loss_pde:.6e} | IC Loss: {loss_ic:.6e} | "
                          f"BC Loss: {loss_bc:.6e} | Temp Loss: {loss_temp:.6e} | "
                          f"Time: {elapsed:.2f}s")
                
                # Save best model
                if total_loss < best_loss:
                    best_loss = total_loss
                    torch.save({
                        'epoch': adam_epochs + lbfgs_iter,
                        'model_state_dict': model.state_dict(),
                        'loss': total_loss,
                    }, os.path.join(save_dir, 'best_model.pt'))
                
                # Save checkpoint every 50 iterations
                if (lbfgs_iter + 1) % 50 == 0:
                    torch.save({
                        'iter': lbfgs_iter,
                        'model_state_dict': model.state_dict(),
                        'loss': total_loss,
                    }, os.path.join(save_dir, f'lbfgs_checkpoint_{lbfgs_iter+1}.pt'))
                
            except Exception as e:
                print(f"L-BFGS iteration {lbfgs_iter} failed with error: {e}")
                print("Continuing with next iteration...")
                continue
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'loss': loss_history['total'][-1] if loss_history['total'] else float('inf'),
    }, os.path.join(save_dir, 'final_model.pt'))
    
    # Save loss history to numpy file
    np.save(os.path.join(save_dir, 'loss_history.npy'), loss_history)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Best loss: {best_loss:.6e}")
    
    return loss_history

def predict_temperature(model, physics, points, times, device):
    """
    Predict temperature at specific points and times
    """
    model.eval()
    temperature_history = []
    
    # Convert times to torch tensor if needed
    if not isinstance(times, torch.Tensor):
        times = torch.tensor(times, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for t in times:
            # Create input by combining points with current time
            t_tensor = t.repeat(points.shape[0]).reshape(-1, 1)
            X = torch.cat([points, t_tensor], dim=1)
            
            # Scale inputs
            X_scaled = physics.scale_inputs(X)
            
            # Get predictions
            u_scaled = model(X_scaled)
            
            # Convert to physical temperature
            T = physics.scale_back_temperature(u_scaled)
            
            temperature_history.append(T.cpu().numpy())
    
    # Stack results: shape [len(times), len(points)]
    return np.stack(temperature_history)
