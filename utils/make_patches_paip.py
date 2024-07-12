import os
import numpy as np
import openslide
from PIL import Image
import glob 
from pathlib import Path
import cv2 as cv

def extract_patches(image_path, patch_size=256, save_path='patches/'):
    # Open the TIFF image using OpenSlide
    slide = cv.imread(image_path)
    
    # Ensure that the save_path directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Get the dimensions of the image
    width, height, _ = slide.shape
    
    # Calculate the number of patches in both dimensions
    num_patches_width = width // patch_size
    num_patches_height = height // patch_size
    
    # Iterate over each patch
    for i in range(num_patches_width):
        for j in range(num_patches_height):
            # Define the coordinates of the current patch
            left = i * patch_size
            upper = j * patch_size
            right = min(left + patch_size, width)
            lower = min(upper + patch_size, height)
            
            # Read the patch as a numpy array
            patch = np.array(slide[lower:upper, left:right,])
            
            # Convert numpy array to PIL image
            patch = Image.fromarray(patch)
            
            # Save the patch as a PNG file
            patch.save(f"{save_path}/patch_{i}_{j}.png")
            print(f"{save_path}/patch_{i}_{j}.png")
            

def get_png_path(base, resolution):
    data_path = base
    image_filenames = []
    mask_filenames = []

    for subdir in os.listdir(data_path):
        subdir_path = os.path.join(data_path, subdir)
        if os.path.isdir(subdir_path):
            image = os.path.join(subdir_path, f"rescaled_image_0_{resolution}x{resolution}.png")
            mask = os.path.join(subdir_path, f"rescaled_mask_0_{resolution}x{resolution}.png")

            # Ensure the image exist
            if os.path.exists(image) and os.path.exists(mask):

                image_filenames.extend([image])
                mask_filenames.extend([mask])
                
    return image_filenames, mask_filenames

def make_patches(path, patch_size=512, save_path="../miccai_patches/"):
    wsi, mask =  get_png_path(path, resolution=8192)
    output_dir = save_path
    os.makedirs(output_dir, exist_ok=True)
    for wsi,mask in zip(wsi, mask):
        wsi_dir = wsi
        mask_dir = mask
        wsi_save_path = os.path.join(save_path, f"{os.path.basename(wsi_dir)}/patches-{patch_size}")
        mask_save_path = os.path.join(save_path, f"{os.path.basename(mask_dir)}/masks-{patch_size}")
        extract_patches(wsi_dir, patch_size=patch_size, save_path=wsi_save_path)
        extract_patches(mask_dir, patch_size=patch_size, save_path=mask_save_path)

    print(f"Done! Totoal {len(wsi)} file.")

if __name__ == "__main__":
    make_patches(path="/lustre/orion/bif146/world-shared/enzhi/paip/output_images_and_masks", 
                 patch_size=512,
                 save_path="/lustre/orion/bif146/world-shared/enzhi/paip_patches/")