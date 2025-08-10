import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import os

from models.models import CNNModel
from data.dataset import EquationDataset
from torch.utils.data import DataLoader
from config.config import EPOCHS, NUM_WORKERS, TRAIN_BATCH_SIZE, TRAINING_SAMPLE_SIZE, TRAINING_IMAGE_SIZE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'output', 'best_model.pth')

def find_polynomial_roots(coeffs, derivative_order=0):
    """Find roots of polynomial or its derivatives using numpy."""
    
    # Coefficients are already in numpy polynomial format (highest degree first)
    poly_coeffs = coeffs
    
    # Take derivatives
    for _ in range(derivative_order):
        if len(poly_coeffs) <= 1:
            return torch.tensor([], device=coeffs.device, dtype=coeffs.dtype)
        
        # Derivative: [6*a6, 5*a5, ..., 1*a1]
        degrees = torch.arange(len(poly_coeffs)-1, 0, -1, device=coeffs.device, dtype=coeffs.dtype)
        poly_coeffs = poly_coeffs[:-1] * degrees
    
    if len(poly_coeffs) <= 1:
        return torch.tensor([], device=coeffs.device, dtype=coeffs.dtype)
    
    # Convert back to numpy for root finding
    poly_coeffs_np = poly_coeffs.detach().cpu().numpy()
    
    try:
        # Find all roots
        roots = np.roots(poly_coeffs_np)
        
        # Filter for real roots in [-5, 5]
        real_roots = []
        for root in roots:
            if np.isreal(root):
                real_val = np.real(root)
                if -5.0 <= real_val <= 5.0:
                    real_roots.append(real_val)
        
        return torch.tensor(real_roots, device=coeffs.device, dtype=coeffs.dtype)
    
    except:
        # If root finding fails, return empty tensor
        return torch.tensor([], device=coeffs.device, dtype=coeffs.dtype)


def compute_special_points_loss(param_pred_denorm, param_labels_raw, x_tensor, x_powers, 
                               crit_weight=5.0, inflection_weight=2.0, gaussian_sigma=0.3):
    """Compute loss that prioritizes critical points and inflection points using a gaussian window"""

    batch_size = param_pred_denorm.shape[0]
    
    total_weighted_loss = 0.0
    total_weight = 0.0
    
    # Base y values (need to flip coeffs for x_powers multiplication)
    y_pred_raw = torch.matmul(param_pred_denorm.flip(-1), x_powers.T)  # (B, N)
    y_true_raw = torch.matmul(param_labels_raw.flip(-1), x_powers.T)   # (B, N)
    
    for b in range(batch_size):
        # Get coefficients for this batch item
        true_coeffs = param_labels_raw[b]   # (7,)
        
        # Find critical points (roots of first derivative)
        true_crit_points = find_polynomial_roots(true_coeffs, derivative_order=1)
        
        # Find inflection points (roots of second derivative)
        true_inflection_points = find_polynomial_roots(true_coeffs, derivative_order=2)
        
        # Combine all special points
        all_special_points = []
        all_weights = []
        
        # Add critical points
        for point in true_crit_points:
            all_special_points.append(point)
            all_weights.append(crit_weight)
        
        # Add inflection points
        for point in true_inflection_points:
            all_special_points.append(point)
            all_weights.append(inflection_weight)
        
        if len(all_special_points) == 0:
            # No special points, use uniform weighting
            weights = torch.ones_like(x_tensor)
        else:
            # Create gaussian weights centered at special points
            weights = torch.ones_like(x_tensor)
            
            for point, point_weight in zip(all_special_points, all_weights):
                # Gaussian window centered at special point
                gaussian_weights = point_weight * torch.exp(
                    -0.5 * ((x_tensor - point) / gaussian_sigma) ** 2
                )
                weights += gaussian_weights
        
        # Compute weighted MSE for this batch item
        y_diff = (y_pred_raw[b] - y_true_raw[b]) ** 2
        weighted_loss = torch.sum(weights * y_diff)
        weight_sum = torch.sum(weights)
        
        total_weighted_loss += weighted_loss
        total_weight += weight_sum
    
    # Average across batch
    if total_weight > 0:
        return total_weighted_loss / total_weight
    else:
        return torch.mean((y_pred_raw - y_true_raw) ** 2)


def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    all_params = []
    for i in range(len(train_loader.dataset)):
        p = train_loader.dataset[i]['parameters']
        if isinstance(p, torch.Tensor):
            p = p.cpu().numpy()
        all_params.append(np.array(p, dtype=np.float32))
    all_params = np.stack(all_params, axis=0)
    param_mean_np = all_params.mean(axis=0)
    param_std_np = all_params.std(axis=0)
    param_std_np = np.maximum(param_std_np, 1e-2)  # floor tiny stds
    param_mean = torch.tensor(param_mean_np, dtype=torch.float32, device=device)
    param_std = torch.tensor(param_std_np, dtype=torch.float32, device=device)
    model.param_mean = param_mean
    model.param_std = param_std

    x_np = np.linspace(-5.0, 5.0, 300)
    x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device)
    x_powers = torch.stack([x_tensor ** i for i in range(7)], dim=1)  # (N,7)

    # hyperparams
    param_weight = 1.0
    y_weight = 40.0
    param_clamp = 2.0
    l2_param_coeff = 1e-3
    
    # New hyperparams for special points
    crit_point_weight = 100.0      # Weight for critical points
    inflection_weight = 50.0      # Weight for inflection points
    gaussian_sigma = 0.4         # Sigma for gaussian windows
    special_points_ratio = 0.99   # How much of y_loss should come from special points

    criterion_reg = nn.SmoothL1Loss()
    criterion_y = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("Param mean:", param_mean_np)
    print("Param std (floored):", param_std_np)
    print(f"Special points weighting: crit={crit_point_weight}, inflection={inflection_weight}, sigma={gaussian_sigma}")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            param_labels_raw = batch['parameters'].to(device).float()

            optimizer.zero_grad()
            param_pred = model(images)  # model returns just param predictions

            param_labels_norm = (param_labels_raw - param_mean) / param_std
            param_pred_bounded = torch.tanh(param_pred) * param_clamp

            loss_reg = criterion_reg(param_pred_bounded, param_labels_norm)

            param_pred_denorm = param_pred_bounded * param_std + param_mean
            
            # Compute standard y loss
            y_pred_raw = torch.matmul(param_pred_denorm, x_powers.T)    # (B, N)
            y_true_raw = torch.matmul(param_labels_raw, x_powers.T)     # (B, N)

            # Normalize y to [-1, 1] by dividing by axis_range (5)
            axis_range = 5.0
            y_pred_norm = y_pred_raw / axis_range
            y_true_norm = y_true_raw / axis_range

            # Standard uniform y loss
            loss_y_uniform = criterion_y(y_pred_norm, y_true_norm)
            
            # Special points weighted loss
            loss_y_special = compute_special_points_loss(
                param_pred_denorm, param_labels_raw, x_tensor, x_powers,
                crit_weight=crit_point_weight, 
                inflection_weight=inflection_weight,
                gaussian_sigma=gaussian_sigma
            )
            
            # Combine uniform and special points losses
            loss_y = ((1 - special_points_ratio) * loss_y_uniform + 
                     special_points_ratio * loss_y_special)

            l2_penalty = l2_param_coeff * torch.mean(param_pred_denorm ** 2)

            total_loss = (param_weight * loss_reg +
                          y_weight * loss_y +
                          l2_penalty)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Total: {total_loss.item():.6f}, "
                      f"Reg: {loss_reg.item():.6f}, Y_uniform: {loss_y_uniform.item():.6f}, "
                      f"Y_special: {loss_y_special.item():.6f}, L2: {l2_penalty.item():.6f}")
                print("  max|denorm_params|:", param_pred_denorm.abs().max().item(),
                      "max|y_pred_raw|:", y_pred_raw.abs().max().item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                param_labels_raw = batch['parameters'].to(device).float()

                param_pred = model(images)
                param_labels_norm = (param_labels_raw - param_mean) / param_std

                param_pred_bounded = torch.tanh(param_pred) * param_clamp
                loss_reg = criterion_reg(param_pred_bounded, param_labels_norm)

                param_pred_denorm = param_pred_bounded * param_std + param_mean

                # Compute y predictions & truth (need to flip coeffs for x_powers multiplication)
                y_pred_raw = torch.matmul(param_pred_denorm.flip(-1), x_powers.T)    # (B, N)
                y_true_raw = torch.matmul(param_labels_raw.flip(-1), x_powers.T)     # (B, N)

                # Normalize y to [-1, 1] by dividing by axis_range (5)
                axis_range = 5.0
                y_pred_norm = y_pred_raw / axis_range
                y_true_norm = y_true_raw / axis_range

                # Standard uniform y loss
                loss_y_uniform = criterion_y(y_pred_norm, y_true_norm)
                
                # Special points weighted loss
                loss_y_special = compute_special_points_loss(
                    param_pred_denorm, param_labels_raw, x_tensor, x_powers,
                    crit_weight=crit_point_weight, 
                    inflection_weight=inflection_weight,
                    gaussian_sigma=gaussian_sigma
                )
                
                # Combine uniform and special points losses
                loss_y = ((1 - special_points_ratio) * loss_y_uniform + 
                         special_points_ratio * loss_y_special)

                l2_penalty = l2_param_coeff * torch.mean(param_pred_denorm ** 2)

                total_loss = (param_weight * loss_reg +
                              y_weight * loss_y +
                              l2_penalty)

                val_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.6f}")
        print(f"                  Val   Loss: {avg_val_loss:.6f}")

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print("  LR reduced to", new_lr)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model


def load_or_train_model():
    """Load existing model or train a new one"""
    model = CNNModel(num_params=7)

    if os.path.exists(MODEL_PATH):
        print("Found existing model! Loading...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            print("✅ Model loaded successfully!")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Training new model instead...")
    else:
        print("No existing model found. Training new model...")
    
    # Train new model
    print("Creating training dataset...")
    train_dataset = EquationDataset(num_samples=TRAINING_SAMPLE_SIZE, image_size=TRAINING_IMAGE_SIZE, split='train')
    print("Creating validation dataset...")
    val_dataset = EquationDataset(num_samples=TRAINING_SAMPLE_SIZE/4, image_size=TRAINING_IMAGE_SIZE, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining model with {total_params:,} parameters")
    
    trained_model = train_model(model, train_loader, val_loader, num_epochs=EPOCHS)
    print("✅ Training complete!")
    
    return trained_model