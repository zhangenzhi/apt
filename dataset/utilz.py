import os
import tifffile
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from multiprocessing import Pool
import re


class XCTSliceCreator:
    """Converts 3D volumes to 2D slices and maintains a manifest file"""
    
    def __init__(self, root_dir, output_dir):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.manifest_path = os.path.join(output_dir, 'slice_manifest.csv')
        
        # Create directory structure
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'meta'), exist_ok=True)

    def _process_single_volume(self, fbp_file, label_file):
        """Process a single volume pair and save all slices"""
        # Load volumes
        fbp_volume = tifffile.imread(os.path.join(self.root_dir, 'FBPs', fbp_file))
        label_volume = tifffile.imread(os.path.join(self.root_dir, 'labels', label_file))
        
        # Ensure 4D: (B, D, H, W)
        if fbp_volume.ndim == 3:
            fbp_volume = np.expand_dims(fbp_volume, axis=0)
        if label_volume.ndim == 3:
            label_volume = np.expand_dims(label_volume, axis=0)
            
        num_slices = fbp_volume.shape[1]
        base_name = fbp_file.split('_reconFBPsimul_')[0]
        volume_id = base_name  # TODO: Extract unique volume ID
        
        slice_records = []
        
        for slice_idx in range(num_slices):
            # Create unique slice ID
            slice_id = f"{volume_id}_s{slice_idx:03d}"
            
            # Create filenames
            img_filename = f"img_{slice_id}.tiff"
            label_filename = f"label_{slice_id}.tiff"
            
            # Save slices with compression
            tifffile.imwrite(
                os.path.join(self.output_dir, 'images', img_filename),
                fbp_volume[0, slice_idx],
                compression='zlib'
            )
            tifffile.imwrite(
                os.path.join(self.output_dir, 'labels', label_filename),
                label_volume[0, slice_idx],
                compression='zlib'
            )
            
            # Store metadata
            slice_records.append({
                'slice_id': slice_id,
                'volume_id': volume_id,
                'original_fbp': fbp_file,
                'original_label': label_file,
                'slice_idx': slice_idx,
                'image_path': os.path.join('images', img_filename),
                'label_path': os.path.join('labels', label_filename)
            })
        print(f"Finished:{fbp_file}/{label_file}")
        return slice_records

    def create_slices(self, num_workers=4):
        """Process all volumes and create manifest file"""
        fbp_dir = os.path.join(self.root_dir, 'FBPs')
        label_dir = os.path.join(self.root_dir, 'labels')
        
        # Find all valid volume pairs
        fbp_files = sorted([f for f in os.listdir(fbp_dir) if f.endswith(('.tiff', '.tif'))])
        valid_pairs = []
        
        for fbp_file in fbp_files:
            base_name = fbp_file.split('_reconFBPsimul_')[0]
            label_file = f"{base_name}_label.tiff"
            if os.path.exists(os.path.join(label_dir, label_file)):
                valid_pairs.append((fbp_file, label_file))
        
        # Process all volumes (parallelized)
        all_records = []
        with Pool(num_workers) as pool:
            results = pool.starmap(self._process_single_volume, valid_pairs)
            for records in results:
                all_records.extend(records)
        
        # Save manifest
        import pandas as pd
        df = pd.DataFrame(all_records)
        df.to_csv(self.manifest_path, index=False)
        
        print(f"Created {len(all_records)} slices from {len(valid_pairs)} volumes")
        return df
    
# Example usage
if __name__ == "__main__":
    # Input directory (original 3D volumes)
    input_dir = "/lustre/orion/lrn075/world-shared/Riken_XCT_Simulated_Data/8192x8192xN_Simulations/Noise_0.05_Blur_2_sparsity_2_NumAng_3600/"
    
    # Output directory for 2D slices
    output_dir = "/lustre/orion/nro108/world-shared/enzhi/Riken_XCT_Simulated_Data/8192x8192_2d_Simulations/Noise_0.05_Blur_2_sparsity_2_NumAng_3600/"
    

    # 1. First create the slices (only need to do this once)
    creator = XCTSliceCreator(
        root_dir=input_dir,
        output_dir=output_dir
    )
    creator.create_slices(num_workers=8)  # Use multiple cores
    

def save_pred_as_mask(pred_tensor, filename):
    """
    Save 5-class prediction tensor as colored mask image
    Args:
        pred_tensor: torch.Size([5, 8192, 8192])
        filename: Output path (.tiff or .png)
    """
    # Convert to class indices (argmax)
    if torch.is_tensor(pred_tensor):
        pred_tensor = pred_tensor.cpu().numpy()  # Convert PyTorch tensor to numpy
    elif not isinstance(pred_tensor, np.ndarray):
        raise TypeError(f"Input must be torch.Tensor or np.ndarray, got {type(pred_tensor)}")
    
    # Validate input shape
    if pred_tensor.ndim != 3 or pred_tensor.shape[0] != 5:
        raise ValueError(f"Expected shape [5, H, W], got {pred_tensor.shape}")
    
    # Convert to class indices (argmax along channel dimension)
    pred_mask = pred_tensor.argmax(axis=0)  # (H, W)
    
    # Create color palette (5 classes + background)
    palette = np.array([
        [0, 0, 0],       # Class 0 - Black
        [255, 0, 0],     # Class 1 - Red
        [0, 255, 0],     # Class 2 - Green
        [0, 0, 255],     # Class 3 - Blue
        [255, 255, 0],   # Class 4 - Yellow
        [255, 0, 255]    # Class 5 - Magenta (if needed)
    ], dtype=np.uint8)
    
    # Apply color mapping
    colored_mask = palette[pred_mask]
    
    # Save based on extension
    if filename.endswith('.tiff') or filename.endswith('.tif'):
        tifffile.imwrite(filename, colored_mask, compression='zlib')
    else:
        Image.fromarray(colored_mask).save(filename)
        
def save_input_as_image(input_tensor, filename):
    """
    Save 1-channel input tensor as TIFF/PNG image
    Args:
        input_tensor: torch.Size([1, 8192, 8192])
        filename: Output path (use .tiff or .png extension)
    """
    # Convert tensor to numpy and squeeze
    if torch.is_tensor(input_tensor):
        img_array = input_tensor.squeeze().cpu().numpy()  # Handle PyTorch tensor
    elif isinstance(input_tensor, np.ndarray):
        img_array = input_tensor.squeeze()  # Handle NumPy array
    else:
        raise TypeError(f"Input must be torch.Tensor or np.ndarray, got {type(input_tensor)}")
    
    # Normalize to 0-255 if needed
    if img_array.dtype != np.uint8:
        img_array = ((img_array - img_array.min()) / 
                    (img_array.max() - img_array.min()) * 255).astype(np.uint8)
    
    # Save based on extension
    if filename.endswith('.tiff') or filename.endswith('.tif'):
        tifffile.imwrite(filename, img_array, compression='zlib')
    else:
        Image.fromarray(img_array).save(filename)