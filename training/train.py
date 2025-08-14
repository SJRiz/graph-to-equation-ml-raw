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

def compute_robust_param_stats(train_loader, percentile_range=(5, 95)):
    """
    Compute parameter statistics using percentiles to handle outliers better.
    This addresses the issue where extreme coefficients are being clipped.
    """
    print("Computing robust parameter statistics...")
    all_params = []
    sample_count = 0

    for batch in train_loader:
        params = batch['parameters'].cpu().numpy()
        all_params.append(params)
        sample_count += params.shape[0]
        if sample_count > 5000:  # Use more samples for better statistics
            break

    all_params = np.concatenate(all_params, axis=0)

    # Use percentiles instead of mean/std to handle outliers
    param_min = np.percentile(all_params, percentile_range[0], axis=0)
    param_max = np.percentile(all_params, percentile_range[1], axis=0)

    # Center and scale based on percentile range
    param_center = (param_min + param_max) / 2
    param_scale = np.maximum((param_max - param_min) / 2, 0.01)

    print(f"Parameter ranges ({percentile_range[0]}th-{percentile_range[1]}th percentile):")
    coeff_names = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    for i, name in enumerate(coeff_names):
        print(f"  {name}: [{param_min[i]:.3f}, {param_max[i]:.3f}] -> center={param_center[i]:.3f}, scale={param_scale[i]:.3f}")

    return (torch.tensor(param_center, dtype=torch.float32),
            torch.tensor(param_scale, dtype=torch.float32))


def train_model(model, train_loader, val_loader, num_epochs=30):
    """
    Training with adaptive bounds that grow during training to prevent saturation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Compute robust parameter statistics
    param_center, param_scale = compute_robust_param_stats(train_loader)
    param_center = param_center.to(device)
    param_scale = param_scale.to(device)

    print(f"Using robust normalization:")
    print(f"  Centers: {param_center}")
    print(f"  Scales: {param_scale}")

    # Loss components
    mse_criterion = nn.MSELoss()
    huber_criterion = nn.HuberLoss(delta=0.5)  # Smaller delta for better precision

    # Optimizer with lower learning rate for stability
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=4)

    inv = 1.0 / (param_scale + 1e-8)
    coeff_weights = (inv / inv.mean()).to(device)   # mean-normalized

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_batches = 0

        # Data-driven base bound: scale by the largest robust scale; allow gentle epoch growth
        max_norm_scale = float(torch.max(param_scale).item())
        base_bound = max(2.0, 3.0 * max_norm_scale)
        bound_factor = min(8.0, base_bound * (1.0 + epoch * 0.02))  # slow growth, higher cap

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            params_true = batch['parameters'].to(device).float()

            # Robust normalization: (x - center) / scale
            params_norm = (params_true - param_center) / (param_scale + 1e-8)

            optimizer.zero_grad()

            # Forward pass
            params_pred_raw = model(images)

            params_pred = torch.tanh(params_pred_raw)

            # Multi-component loss with coefficient importance
            mse_loss = mse_criterion(params_pred, params_norm)
            huber_loss = huber_criterion(params_pred, params_norm)

            # Weighted MSE (importance per coefficient)
            weighted_mse = torch.mean(coeff_weights * (params_pred - params_norm) ** 2)

            # encourage predicted magnitude to match data magnitude
            pred_mag = torch.mean(torch.abs(params_pred))
            target_mag = torch.mean(torch.abs(params_norm)).detach().clamp(min=1e-6)
            magnitude_loss = (pred_mag - target_mag) ** 2

            # zero-coefficient penalty (in normalized space)
            zero_tol = 1e-6
            zero_mask = (torch.abs(params_norm) < zero_tol).float()
            zero_penalty = torch.mean(zero_mask * (params_pred ** 2))

            # Total loss
            total_loss = (
                0.4 * mse_loss
                + 0.3 * huber_loss
                + 0.2 * weighted_mse
                + 0.01 * magnitude_loss
                + 0.005 * zero_penalty
            )

            # L2 regularization
            l2_reg = 0.0005 * torch.mean(params_pred_raw ** 2)
            total_loss = total_loss + l2_reg

            total_loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)

            optimizer.step()

            # Metrics
            train_loss += total_loss.item()
            train_mse += mse_loss.item()
            train_batches += 1

            # logging
            if batch_idx % 150 == 0:
                # Denormalize for interpretability
                params_pred_denorm = params_pred * (param_scale + 1e-8) + param_center

                # saturation diagnostics
                pred_abs = torch.abs(params_pred)
                pred_range = pred_abs.max()
                saturation_pct = (pred_range / bound_factor * 100).item()
                saturated = (pred_abs > 0.99 * bound_factor).float()
                saturation_frac = saturated.mean().item()

                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss={total_loss.item():.6f}")
                print(f"  MSE: {mse_loss.item():.4f}, Huber: {huber_loss.item():.4f}")
                print(f"  Bound factor: {bound_factor:.2f}, Pred mag: {pred_mag.item():.3f}")
                print(f"  Grad norm: {grad_norm:.3f}")

                # Show sample comparison
                true_sample = params_true[0][:]
                pred_sample = params_pred_denorm[0][:]
                print(f"  Sample 0 True: {true_sample.detach().cpu().numpy()}")
                print(f"  Sample 0 Pred: {pred_sample.detach().cpu().numpy()}")

                print(f"  Max prediction: {pred_range.item():.2f}/{bound_factor:.2f} ({saturation_pct:.1f}%)")
                print(f"  Saturation fraction (coeffs near bound): {saturation_frac:.3f}")

                if saturation_pct > 90 or saturation_frac > 0.25:
                    print(f"High saturation!")

        # Validation
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                params_true = batch['parameters'].to(device).float()
                params_norm = (params_true - param_center) / (param_scale + 1e-8)

                params_pred_raw = model(images)
                params_pred = torch.tanh(params_pred_raw)

                mse_loss = mse_criterion(params_pred, params_norm)
                huber_loss = huber_criterion(params_pred, params_norm)
                weighted_mse = torch.mean(coeff_weights * (params_pred - params_norm) ** 2)

                pred_mag = torch.mean(torch.abs(params_pred))
                target_mag = torch.mean(torch.abs(params_norm)).clamp(min=1e-6)
                magnitude_loss = (pred_mag - target_mag) ** 2

                zero_tol = 1e-6
                zero_mask = (torch.abs(params_norm) < zero_tol).float()
                zero_penalty = torch.mean(zero_mask * (params_pred ** 2))

                l2_reg = 0.0005 * torch.mean(params_pred_raw ** 2)

                total_loss = (
                    0.4 * mse_loss
                    + 0.3 * huber_loss
                    + 0.2 * weighted_mse
                    + 0.01 * magnitude_loss
                    + 0.05 * zero_penalty
                    + l2_reg
                )

                val_loss += total_loss.item()
                val_mse += mse_loss.item()
                val_batches += 1

        avg_train_loss = train_loss / max(1, train_batches)
        avg_val_loss = val_loss / max(1, val_batches)
        avg_train_mse = train_mse / max(1, train_batches)
        avg_val_mse = val_mse / max(1, val_batches)

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f} (MSE: {avg_train_mse:.6f})")
        print(f"  Val Loss:   {avg_val_loss:.6f} (MSE: {avg_val_mse:.6f})")
        print(f"  Bound factor: {bound_factor:.2f}")

        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            try:
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  New best model saved! Val loss: {avg_val_loss:.6f}")
            except Exception as e:
                print("ERROR SAVING BEST MODEL:", str(e))
        else:
            patience_counter += 1

        if patience_counter >= 8:
            print("Early stopping triggered")
            break

    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
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