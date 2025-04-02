import os
import tifffile
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