import os
import sys
sys.path.append("./")
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn as nn

import tifffile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataset.utilz import save_input_as_image,save_pred_as_mask
from model.unet import Unet
import cv2

# Set the flag to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(id=32):
    sample_slice_path =  f"/lustre/orion/nro108/world-shared/enzhi/spring8data/8192_output_2/No_020/0{id}"
    num_sample_slice = len(os.listdir(sample_slice_path))
    image_filenames = []
    for i in range(num_sample_slice):
        # Ensure the image exist
        img_name = f"volume_{str(i).zfill(3)}.raw"
        image = os.path.join(sample_slice_path, img_name)
        if os.path.exists(image):
            image_filenames.extend([image])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    num_class = 5
    model = Unet(n_class=num_class, in_channels=1, pretrained=False)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join("/lustre/orion/nro108/world-shared/enzhi/apt/unet-s8d-n32-dce", "best_score_model.pth")))
    
    # import pdb;pdb.set_trace()
    
    pred_slices = []
    image_slices = []
    for idx,img_name in enumerate(image_filenames):
        image = np.fromfile(img_name, dtype=np.uint16).reshape([8192, 8192, 1])
        image = (image[:] / 255).astype(np.uint8)
        image = torch.Tensor(image)
        image = (image - image.min()) / (image.max() - image.min()+1e-4)
        image = image.permute(2,0,1).unsqueeze(0)
        # save_input_as_image(image[0].permute(1,2,0), f"real_img_{idx}.png")
        
        with torch.no_grad():
            image = image.to(device=device)
            pred = model(image)
            # save_pred_as_mask(pred[0], f"pred_{idx}.png")
            
            pred = pred[0].argmax(axis=0).unsqueeze(-1)
            image = image[0].permute(1,2,0)
            pred_resized = cv2.resize(pred.cpu().numpy(), (512, 512), interpolation=cv2.INTER_NEAREST)
            image_resized = cv2.resize(image.cpu().numpy(), (512, 512), interpolation=cv2.INTER_NEAREST)
            pred_slices.append(pred_resized)
            image_slices.append(image_resized)
    
    pred_slices = np.stack(pred_slices, axis=0)    # (N, 512, 512)
    image_slices = np.stack(image_slices, axis=0) # (N, 512, 512)
    np.savez(f"output_3d_data_{id}.npz", dem=pred_slices, image=image_slices)
    print("Saved as 3D data:", pred_slices.shape, image_slices.shape)

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation

def post_process(sample_id):
    # 1. Load the NPZ file
    data = np.load(f"output_3d_data_{sample_id}.npz")

    # 2. Extract arrays
    dem = data["dem"]      # Shape: (N, H, W)
    image = data["image"]  # Shape: (N, H, W)

    # Process DEM to create binary mask (3 -> 1, others -> 0)
    mask = np.where(dem == 3, 1, 0).astype(np.float32)
    
    # Apply mask to image (element-wise multiplication)
    masked_image = image * mask
    
    # Convert to float32
    masked_image = masked_image.astype(np.float32)
    
    # Handle empty frames by interpolating
    for idx in range(160):
        if np.sum(masked_image[idx]) <= 100:
            masked_image[idx] = masked_image[idx-1]
            
    # Fill small holes in the mask using morphological closing
    # This will connect nearby regions and fill small gaps
    for idx in range(mask.shape[0]):
        # Create a binary mask for this slice
        slice_mask = mask[idx]
        
        # Use binary closing to fill small holes
        closed_mask = binary_closing(slice_mask, structure=np.ones((3,3)), iterations=1)
        
        # Apply the closed mask to the image
        masked_image[idx] = masked_image[idx] * slice_mask  # Keep original masked areas
        # filled_values = image[idx] * (closed_mask - slice_mask)  # Get values from newly filled areas
        filled_values = closed_mask - slice_mask  # Get values from newly filled areas
        masked_image[idx] = masked_image[idx] + filled_values  # Combine them
    
    masked_image = masked_image * 1024
    
    # 3. Save each array as raw binary file
    def save_as_raw(array, filename):
        with open(filename, "wb") as f:
            array.flatten().tofile(f)

    save_as_raw(mask, f"dem_mask_{sample_id}.raw")
    save_as_raw(masked_image, f"masked_image_{sample_id}.raw")

    print("Saved raw files:")
    print(f"dem_mask.raw     - Shape: {mask.shape}, Dtype: {mask.dtype}")
    print(f"masked_image.raw - Shape: {masked_image.shape}, Dtype: {masked_image.dtype}")
    
    # Verification print
    unique_values = np.unique(mask)
    print(f"Unique values in mask: {unique_values}")
    print(f"Mask value counts: 1: {np.sum(mask == 1)}, 0: {np.sum(mask == 0)}")

if __name__ == "__main__":
    for i in range(27,32):
        main(id=i)
    # post_process(sample_id=34)