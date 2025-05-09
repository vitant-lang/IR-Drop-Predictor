import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score
from tqdm import tqdm
from model import InterposerUNet, ChipletUNet
from custom_dataset import CustomDataset
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

# Initialize configuration
current_dir = os.path.dirname(os.path.abspath(__file__))

# Common configuration
common_config = {
    'train_data_dir': os.path.join(current_dir, 'circuit_train'),
    'val_data_dir': os.path.join(current_dir, 'circuit_valid'),
    'model_save_dir': os.path.join(current_dir, 'Weight'),
    'log_dir': os.path.join(current_dir, 'runs'),
    'batch_size': 1,
    'num_epochs': 5,
    'patience': 15,
    'num_workers': 0,
    'save_interval': 5,
    'threshold': 0.3
}

# Model-specific configurations
model_configs = {
    'interposer': {
        'learning_rate': 5e-4,
        'input_channels': 5,
        'target_index': 1  # interposer_ir_drop is targets[:,1]
    },
    'chiplet': {
        'learning_rate': 1e-3,
        'input_channels': 3,
        'target_index': 0  # chiplet_ir_drop is targets[:,0]
    }
}

class EnhancedWeightedLoss(nn.Module):
    """Enhanced weighted loss function, configurable for different models"""
    def __init__(self, model_type='interposer', alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.model_type = model_type
        
    def forward(self, pred, target):
        # Base weight
        base_weight = (target > 0).float() * 9 + 1
        
        if self.model_type == 'interposer':
            # Smooth area weights for Interposer
            smooth_mask = 1 + torch.exp(-target*5)
            weight = base_weight * smooth_mask
        else:
            # Enhanced edge weights for Chiplet
            edge_mask = self._detect_edges(target) * 3 + 1
            weight = base_weight * edge_mask
            
        return torch.mean(weight*(pred - target)**2)
    
    def _detect_edges(self, x):
        """Edge detection"""
        sobel_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], 
                             dtype=torch.float32, device=x.device)
        sobel_y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]],
                             dtype=torch.float32, device=x.device)
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        return (grad_x.abs() + grad_y.abs()).clamp(0, 1)

def calculate_metrics(pred, target, threshold):
    """Calculate comprehensive metrics including F1 and IoU"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    TP = (pred_binary * target_binary).sum()
    FP = (pred_binary * (1 - target_binary)).sum()
    FN = ((1 - pred_binary) * target_binary).sum()
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    
    return {
        'f1': f1.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

class CustomDatasetWrapper:
    """Wrapper for dataset to handle different model inputs/outputs"""
    def __init__(self, dataset, model_type):
        self.dataset = dataset
        self.model_type = model_type
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        inputs, targets = self.dataset[idx]
        if self.model_type == 'chiplet':
            return inputs[:3], targets[0:1]  # First 3 channels, chiplet target
        else:
            return inputs, targets[1:2]  # All 5 channels, interposer target

def train_model(model_type='interposer'):
    """Train specified model type"""
    # Merge configurations
    config = {**common_config,**model_configs[model_type]}
    
    # Initialize paths
    os.makedirs(config['model_save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    if model_type == 'interposer':
        model = InterposerUNet().to(device)
    else:
        model = ChipletUNet().to(device)
    
    # Loss function and optimizer
    criterion = EnhancedWeightedLoss(model_type=model_type)
    optimizer = optim.AdamW(model.parameters(), 
                         lr=config['learning_rate'],
                         weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=5, factor=0.5, verbose=True)
    
    # Logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(config['log_dir'], f'{model_type}_{timestamp}'))
    
    # Dataset and data loaders
    train_dataset = CustomDatasetWrapper(CustomDataset(config['train_data_dir']), model_type)
    val_dataset = CustomDatasetWrapper(CustomDataset(config['val_data_dir']), model_type)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Training loop
    best_metrics = {'f1': 0, 'iou': 0}
    epochs_no_improve = 0
    scaler = GradScaler()
    
    for epoch in range(config['num_epochs']):
        # Training phase
        torch.cuda.empty_cache()
        model.train()
        
        train_metrics = {'loss': 0, 'f1': 0}
        
        train_loop = tqdm(train_loader, desc=f'Train Epoch {epoch+1}', leave=False)
        for inputs, targets in train_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Train model
            with autocast():
                output = model(inputs)
                loss = criterion(output, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Calculate training metrics
            with torch.no_grad():
                metrics = calculate_metrics(output, targets, config['threshold'])
                train_metrics['loss'] += loss.item()
                train_metrics['f1'] += metrics['f1']
            
            # Update progress bar
            train_loop.set_postfix({
                'Loss': loss.item(),
                'F1': metrics['f1']
            })
        
        # Calculate average training metrics
        for metric in train_metrics:
            train_metrics[metric] /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_metrics = {'loss': 0, 'f1': 0, 'iou': 0}
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc='Validating', leave=False)
            for inputs, targets in val_loop:
                inputs, targets = inputs.to(device), targets.to(device)
                
                output = model(inputs)
                loss = criterion(output, targets)
                metrics = calculate_metrics(output, targets, config['threshold'])
                
                val_metrics['loss'] += loss.item()
                val_metrics['f1'] += metrics['f1']
                val_metrics['iou'] += metrics['iou']
                
                val_loop.set_postfix({
                    'F1': metrics['f1']
                })
        
        # Calculate average validation metrics
        for metric in val_metrics:
            val_metrics[metric] /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_metrics['f1'])
        
        # Log metrics
        writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        
        writer.add_scalars('F1', {
            'train': train_metrics['f1'],
            'val': val_metrics['f1']
        }, epoch)
        
        writer.add_scalar('IoU', val_metrics['iou'], epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch information
        print(f'\nEpoch [{epoch+1}/{config["num_epochs"]}] - {model_type}')
        print('Train Loss: {:.4f}, Val Loss: {:.4f}, Val F1: {:.4f}, IoU: {:.4f}'.format(
            train_metrics['loss'],
            val_metrics['loss'],
            val_metrics['f1'],
            val_metrics['iou']))
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0 or epoch == config['num_epochs'] - 1:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': val_metrics,
                'config': config
            }
            
            checkpoint_path = os.path.join(
                config['model_save_dir'], 
                f'{model_type}_checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved to: {checkpoint_path}')
        
        # Save best model
        if val_metrics['f1'] > best_metrics['f1']:
            best_metrics = val_metrics
            torch.save(model.state_dict(), 
                     os.path.join(config['model_save_dir'], f'best_{model_type}_model.pth'))
            print(f'New best model, F1: {val_metrics["f1"]:.4f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['patience']:
                print(f'\nEarly stopping triggered, no improvement for {config["patience"]} epochs')
                break
    
    writer.close()
    print(f'\n{model_type} training completed')
    print('Best metrics:')
    print(f'  F1: {best_metrics["f1"]:.4f}')
    print(f'  IoU: {best_metrics["iou"]:.4f}')

if __name__ == '__main__':
    # Train Interposer model
    train_model(model_type='interposer')
    
    # Train Chiplet model
    train_model(model_type='chiplet')