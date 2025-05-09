import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from torch.utils.data import DataLoader
import os
import subprocess
import platform
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# Get base directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dynamic model imports
try:
    from model import InterposerUNet as InterposerModel
    from model import ChipletUNet as ChipletModel
except ImportError:
    raise ImportError("Could not import model classes, please ensure correct model definitions")

from custom_dataset import CustomDataset

# Configure paths
LOCAL_PATHS = {
    'model_weights': os.path.join(BASE_DIR, 'Weight'),
    'test_data': os.path.join(BASE_DIR, 'circuit_test'),
    'output_viz': os.path.join(BASE_DIR, 'Output', 'viz_checkpoint'),
    'output_pred': os.path.join(BASE_DIR, 'Output', 'pred_results_checkpoint'),
    'metrics_log': os.path.join(BASE_DIR, 'Output', 'metrics_checkpoint')
}

def ensure_directories():
    """Create all necessary output directories"""
    for path in LOCAL_PATHS.values():
        os.makedirs(path, exist_ok=True)

def plot_comparison(pred, target, title, filename):
    """Plot and save comparison between prediction and target (heatmaps)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)
    
    # Set shared color range
    combined_data = np.concatenate([pred.flatten(), target.flatten()])
    vmin = np.percentile(combined_data, 5)
    vmax = np.percentile(combined_data, 95)
    
    # Prediction heatmap
    im1 = axes[0].imshow(pred, cmap='viridis', norm=colors.Normalize(vmin=vmin, vmax=vmax))
    axes[0].set_title('Predicted', fontsize=14)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Target heatmap
    im2 = axes[1].imshow(target, cmap='viridis', norm=colors.Normalize(vmin=vmin, vmax=vmax))
    axes[1].set_title('Target', fontsize=14)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Error heatmap
    error = np.abs(pred - target)
    im3 = axes[2].imshow(error, cmap='hot', norm=colors.Normalize(vmin=0, vmax=np.max(error)))
    axes[2].set_title('Absolute Error', fontsize=14)
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    save_path = os.path.join(LOCAL_PATHS['output_viz'], filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap comparison saved to: {save_path}")
    return save_path

def plot_curve_comparison(pred, target, title, filename):
    """Plot and save curve comparison between prediction and target"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get center row data
    center_row = pred.shape[0] // 2
    pred_line = pred[center_row, :]
    target_line = target[center_row, :]
    x_axis = np.arange(len(pred_line))
    
    # Plot curves
    ax.plot(x_axis, pred_line, 'b-', linewidth=2, label='Predicted')
    ax.plot(x_axis, target_line, 'r--', linewidth=2, label='Target')
    
    # Fill error regions
    ax.fill_between(x_axis, pred_line, target_line, where=(pred_line > target_line), 
                   facecolor='red', alpha=0.3, interpolate=True, label='Overestimation')
    ax.fill_between(x_axis, pred_line, target_line, where=(pred_line <= target_line), 
                   facecolor='blue', alpha=0.3, interpolate=True, label='Underestimation')
    
    # Set plot properties
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(LOCAL_PATHS['output_viz'], f"curve_{filename}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Curve comparison saved to: {save_path}")
    return save_path

def calculate_metrics(pred, target):
    """Calculate various evaluation metrics"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Handle NaN and infinite values
    valid_mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
    if not np.all(valid_mask):
        print(f"Warning: Found {len(pred_flat) - np.sum(valid_mask)} invalid values")
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        
        if len(pred_flat) == 0:
            return {
                'mse': np.nan,
                'mae': np.nan,
                'max_error': np.nan,
                'relative_error': np.nan
            }
    
    metrics = {
        'mse': mean_squared_error(target_flat, pred_flat),
        'mae': mean_absolute_error(target_flat, pred_flat),
        'max_error': np.max(np.abs(target_flat - pred_flat)),
        'relative_error': np.mean(np.abs(target_flat - pred_flat) / (np.abs(target_flat) + 1e-10))
    }
    return metrics

def save_metrics(metrics_dict, filename):
    """Save metrics to CSV"""
    df = pd.DataFrame.from_dict(metrics_dict, orient='index').T
    save_path = os.path.join(LOCAL_PATHS['metrics_log'], filename)
    df.to_csv(save_path, index=False, float_format='%.6f')
    print(f"Metrics saved to: {save_path}")

def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint file"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Successfully loaded checkpoint: {checkpoint_path}")
    
    # Return epoch info if available
    return checkpoint.get('epoch', 'unknown')

def evaluate_checkpoints():
    """Main function for checkpoint evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Initialize models
    interposer_model = InterposerModel().to(device)
    chiplet_model = ChipletModel().to(device)
    
    # Define checkpoints to evaluate
    checkpoints = {
        'interposer': os.path.join(LOCAL_PATHS['model_weights'], 'interposer_checkpoint_epoch_5.pth'),
        'chiplet': os.path.join(LOCAL_PATHS['model_weights'], 'chiplet_checkpoint_epoch_5.pth')
    }
    
    # Load checkpoints
    interposer_epoch = load_checkpoint(interposer_model, checkpoints['interposer'], device)
    chiplet_epoch = load_checkpoint(chiplet_model, checkpoints['chiplet'], device)
    
    # Load test data
    test_dataset = CustomDataset(LOCAL_PATHS['test_data'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize metrics storage
    global_metrics = {
        'interposer': {'mse': [], 'mae': [], 'max_error': [], 'relative_error': []},
        'chiplet': {'mse': [], 'mae': [], 'max_error': [], 'relative_error': []}
    }
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Prepare inputs
            interposer_input = inputs  # 5-channel input
            chiplet_input = inputs[:, :3]  # Only first 3 channels
            
            # Predict
            interposer_pred = interposer_model(interposer_input).cpu().numpy()[0, 0]
            chiplet_pred = chiplet_model(chiplet_input).cpu().numpy()[0, 0]
            
            # Get ground truth
            chiplet_target = targets[:, 0].cpu().numpy()[0]
            interposer_target = targets[:, 1].cpu().numpy()[0]
            
            # Save raw data
            np.save(os.path.join(LOCAL_PATHS['output_pred'], f'interposer_pred_{i}.npy'), interposer_pred)
            np.save(os.path.join(LOCAL_PATHS['output_pred'], f'chiplet_pred_{i}.npy'), chiplet_pred)
            np.save(os.path.join(LOCAL_PATHS['output_pred'], f'interposer_target_{i}.npy'), interposer_target)
            np.save(os.path.join(LOCAL_PATHS['output_pred'], f'chiplet_target_{i}.npy'), chiplet_target)
            
            # Generate comparison plots
            interposer_heatmap_path = plot_comparison(
                interposer_pred, interposer_target,
                f'Interposer IR Drop (Epoch {interposer_epoch}) - Case {i}',
                f'interposer_comparison_case{i}.png'
            )
            
            chiplet_heatmap_path = plot_comparison(
                chiplet_pred, chiplet_target,
                f'Chiplet IR Drop (Epoch {chiplet_epoch}) - Case {i}',
                f'chiplet_comparison_case{i}.png'
            )
            
            interposer_curve_path = plot_curve_comparison(
                interposer_pred, interposer_target,
                f'Interposer IR Drop Curve (Epoch {interposer_epoch}) - Case {i}',
                f'interposer_curve_case{i}.png'
            )
            
            chiplet_curve_path = plot_curve_comparison(
                chiplet_pred, chiplet_target,
                f'Chiplet IR Drop Curve (Epoch {chiplet_epoch}) - Case {i}',
                f'chiplet_curve_case{i}.png'
            )
            
            # Calculate metrics
            interposer_metrics = calculate_metrics(interposer_pred, interposer_target)
            chiplet_metrics = calculate_metrics(chiplet_pred, chiplet_target)
            
            # Store metrics
            for key in interposer_metrics:
                global_metrics['interposer'][key].append(interposer_metrics[key])
                global_metrics['chiplet'][key].append(chiplet_metrics[key])
            
            # Print results
            print(f"\nTest Case {i+1}/{len(test_loader)}:")
            print(f"Interposer (Epoch {interposer_epoch}) - MSE: {interposer_metrics['mse']:.4e}")
            print(f"           MAE: {interposer_metrics['mae']:.4e}, MaxErr: {interposer_metrics['max_error']:.4e}")
            print(f"Chiplet (Epoch {chiplet_epoch}) - MSE: {chiplet_metrics['mse']:.4e}")
            print(f"         MAE: {chiplet_metrics['mae']:.4e}, MaxErr: {chiplet_metrics['max_error']:.4e}")
    
    # Calculate average metrics
    avg_metrics = {
        'interposer': {k: np.nanmean(v) for k, v in global_metrics['interposer'].items()},
        'chiplet': {k: np.nanmean(v) for k, v in global_metrics['chiplet'].items()}
    }
    
    # Save global metrics
    save_metrics(avg_metrics, 'checkpoint_metrics_summary.csv')
    
    # Print final results
    print("\n=== Checkpoint Evaluation Results ===")
    print(f"Interposer Model (Epoch {interposer_epoch}):")
    print(f"  MSE: {avg_metrics['interposer']['mse']:.4e}")
    print(f"  MAE: {avg_metrics['interposer']['mae']:.4e}")
    print(f"  Max Error: {avg_metrics['interposer']['max_error']:.4e}")
    print(f"  Relative Error: {avg_metrics['interposer']['relative_error']:.4f}")
    
    print(f"\nChiplet Model (Epoch {chiplet_epoch}):")
    print(f"  MSE: {avg_metrics['chiplet']['mse']:.4e}")
    print(f"  MAE: {avg_metrics['chiplet']['mae']:.4e}")
    print(f"  Max Error: {avg_metrics['chiplet']['max_error']:.4e}")
    print(f"  Relative Error: {avg_metrics['chiplet']['relative_error']:.4f}")

if __name__ == '__main__':
    # Ensure all directories exist
    ensure_directories()
    
    # Print configuration info
    print("=== Checkpoint Evaluation Configuration ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {BASE_DIR}")
    print(f"Test data path: {LOCAL_PATHS['test_data']}")
    print(f"Model weights path: {LOCAL_PATHS['model_weights']}")
    print(f"Visualization output path: {LOCAL_PATHS['output_viz']}")
    print(f"Prediction output path: {LOCAL_PATHS['output_pred']}")
    print(f"Metrics log path: {LOCAL_PATHS['metrics_log']}")
    
    # Run evaluation
    print("\nStarting checkpoint evaluation...")
    evaluate_checkpoints()
    print("\nEvaluation completed! All outputs saved to:")
    print(f"- Visualization results: {LOCAL_PATHS['output_viz']}")
    print(f"- Prediction data: {LOCAL_PATHS['output_pred']}")
    print(f"- Evaluation metrics: {LOCAL_PATHS['metrics_log']}")