import os
import tifffile
import torch
import numpy as np
from tqdm import tqdm

def save_3d_as_2d_slices(root_dir, output_dir, max_workers=4):
    """
    Convert 3D volumes to 2D slices and save them in an organized directory structure.
    
    Args:
        root_dir: Path to the directory containing FBPs and labels folders
        output_dir: Path where to save the 2D slices
        max_workers: Number of parallel workers for processing
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # Get all volume pairs
    fbp_dir = os.path.join(root_dir, 'FBPs')
    label_dir = os.path.join(root_dir, 'labels')
    
    fbp_files = [f for f in os.listdir(fbp_dir) if f.endswith(('.tiff', '.tif'))]
    valid_pairs = []
    
    for fbp_file in fbp_files:
        base_name = fbp_file.split('_reconFBPsimul_')[0]
        label_file = f"{base_name}_label.tiff"
        if os.path.exists(os.path.join(label_dir, label_file)):
            valid_pairs.append((fbp_file, label_file))
    
    # Process each volume pair
    for fbp_file, label_file in tqdm(valid_pairs, desc="Processing volumes"):
        import pdb;pdb.set_trace()
        
        # Load volumes
        fbp_volume = tifffile.imread(os.path.join(fbp_dir, fbp_file))
        label_volume = tifffile.imread(os.path.join(label_dir, label_file))
        
        # Ensure we have 4D tensors (B, D, H, W)
        if fbp_volume.ndim == 3:
            fbp_volume = np.expand_dims(fbp_volume, axis=0)
        if label_volume.ndim == 3:
            label_volume = np.expand_dims(label_volume, axis=0)
        
        # Get volume info
        num_slices = fbp_volume.shape[1]
        vol_name = os.path.splitext(fbp_file)[0]
        
        # Save each slice
        for slice_idx in range(num_slices):
            # Get 2D slices
            img_slice = fbp_volume[0, slice_idx]
            label_slice = label_volume[0, slice_idx]
            
            # Create filenames
            img_filename = f"{vol_name}_slice{slice_idx:03d}.tiff"
            label_filename = f"{base_name}_label_slice{slice_idx:03d}.tiff"
            
            # Save slices
            tifffile.imwrite(
                os.path.join(output_dir, 'images', img_filename),
                img_slice,
                compression='zlib'  # Lossless compression to save space
            )
            tifffile.imwrite(
                os.path.join(output_dir, 'labels', label_filename),
                label_slice,
                compression='zlib'
            )

# Example usage
if __name__ == "__main__":
    # Input directory (original 3D volumes)
    input_dir = "/lustre/orion/lrn075/world-shared/lrn075/Riken_XCT_Simulated_Data/8192x8192xN_Simulations/Noise_0.05_Blur_2_sparsity_2_NumAng_3600/"
    
    # Output directory for 2D slices
    output_dir = "/lustre/orion/nro108/world-shared/enzhi/Riken_XCT_Simulated_Data/8192x8192_2d_Simulations/Noise_0.05_Blur_2_sparsity_2_NumAng_3600/"
    
    # Process and save all slices
    save_3d_as_2d_slices(input_dir, output_dir)