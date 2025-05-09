import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
import cv2

class CustomDataset(Dataset):
    def __init__(self, root_dir, target_size=(2000, 2000), transform=None):
        """
        Complete dataset class with automatic handling of inconsistent sizes
        
        Parameters:
            root_dir: Root directory of data
            target_size: Target size for uniform resizing (height, width)
            transform: Optional data augmentation transforms
        """
        self.root_dir = os.path.normpath(root_dir)
        self.target_size = target_size
        self.transform = transform
        self.data_folders = self._get_valid_folders()
        
        if not self.data_folders:
            raise RuntimeError(f"No valid test case folders found in {self.root_dir}")
        
        # Precompute normalization parameters
        self._compute_normalization_params()
        
        # Print dataset info
        print(f"Successfully initialized dataset with {len(self.data_folders)} samples")
        print(f"Target size: {target_size}")
    
    def _get_valid_folders(self):
        """Get all test case folders containing complete files"""
        folders = []
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path) and folder.startswith("testcase"):
                # Check if all required files exist
                required_files = [
                    "current.csv",
                    "chiplet_eff_dist.csv",
                    "chiplet_pdn_density.csv",
                    "interposer_eff_dist.csv",
                    "interposer_density.csv",
                    "chiplet_ir_drop.csv",
                    "interposer_ir_drop.csv"
                ]
                
                if all(os.path.exists(os.path.join(folder_path, f)) for f in required_files):
                    folders.append(folder)
                else:
                    print(f"Warning: Skipping folder {folder} - missing required files")
        
        return sorted(folders)

    def _compute_normalization_params(self):
        """Precompute global normalization parameters for input data"""
        input_mins = []
        input_maxs = []
        
        # Check 10 samples
        for folder in self.data_folders[:10]:
            inputs, _ = self._load_and_resize(folder, normalize=False)
            input_mins.append(np.min(inputs, axis=(1, 2)))
            input_maxs.append(np.max(inputs, axis=(1, 2)))
        
        # Compute global min/max
        self.global_input_min = np.min(input_mins, axis=0).reshape(5, 1, 1)
        self.global_input_max = np.max(input_maxs, axis=0).reshape(5, 1, 1)
        
        # Avoid division by zero
        self.global_input_max[self.global_input_max == self.global_input_min] = 1
        self.global_input_min[self.global_input_max == self.global_input_min] = 0
        
        print("Global normalization parameters computed")
        print(f"Input min values: {np.squeeze(self.global_input_min)}")
        print(f"Input max values: {np.squeeze(self.global_input_max)}")

    def _load_and_resize(self, folder_name, normalize=True):
        """Load data and resize to uniform dimensions"""
        folder_path = os.path.join(self.root_dir, folder_name)
        
        # Load input data
        input_data = []
        for f in [
            "current.csv",
            "chiplet_eff_dist.csv",
            "chiplet_pdn_density.csv",
            "interposer_eff_dist.csv",
            "interposer_density.csv"
        ]:
            file_path = os.path.join(folder_path, f)
            data = pd.read_csv(file_path, header=None).values.astype(np.float32)
            
            # Apply Gaussian smoothing to sparse data
            if 'chiplet' in f or 'current' in f:
                data = gaussian_filter(data, sigma=1)
            
            # Resize
            if data.shape != self.target_size:
                data = cv2.resize(data, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
            
            input_data.append(data)
        
        # Stack inputs (5, H, W)
        inputs = np.stack(input_data, axis=0)
        
        # Load output data
        output_data = []
        for f in ["chiplet_ir_drop.csv", "interposer_ir_drop.csv"]:
            file_path = os.path.join(folder_path, f)
            data = pd.read_csv(file_path, header=None).values.astype(np.float32)
            
            # Resize
            if data.shape != self.target_size:
                data = cv2.resize(data, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
            
            output_data.append(data)
        
        # Stack outputs (2, H, W)
        outputs = np.stack(output_data, axis=0)
        
        # Apply normalization
        if normalize:
            inputs = (inputs - self.global_input_min) / (self.global_input_max - self.global_input_min + 1e-10)
            
            # Independent output normalization
            min_out = np.min(outputs, axis=(1, 2), keepdims=True)
            max_out = np.max(outputs, axis=(1, 2), keepdims=True)
            max_out[max_out == min_out] = 1
            outputs = (outputs - min_out) / (max_out - min_out + 1e-10)
        
        return inputs, outputs

    def __len__(self):
        return len(self.data_folders)

    def __getitem__(self, index):
        """Get data item, ensuring tensor return"""
        inputs, outputs = self._load_and_resize(self.data_folders[index])
        
        # Convert to tensors
        inputs_tensor = torch.from_numpy(inputs.copy()).float()
        outputs_tensor = torch.from_numpy(outputs.copy()).float()
        
        if self.transform:
            inputs_tensor = self.transform(inputs_tensor)
            outputs_tensor = self.transform(outputs_tensor)
        
        return inputs_tensor, outputs_tensor