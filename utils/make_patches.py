import os
import numpy as np
import openslide
from PIL import Image
import glob 
from pathlib import Path

def extract_patches(image_path, patch_size=256, save_path='patches/'):
    # Open the TIFF image using OpenSlide
    slide = openslide.OpenSlide(image_path)
    
    # Ensure that the save_path directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Get the dimensions of the image
    width, height = slide.dimensions
    
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
            patch = np.array(slide.read_region((left, upper), 0, (right-left, lower-upper)))
            
            # Convert numpy array to PIL image
            patch = Image.fromarray(patch)
            
            # Save the patch as a PNG file
            patch.save(f"{save_path}/patch_{i}_{j}.png")
            print(f"{save_path}/patch_{i}_{j}.png")
            

def get_tiff_path(datapath):
    image_files = []
    filenames = []
    for f in glob.glob(os.path.join(datapath, "*_wsi.tiff")):
        image_files.append(f)
        filenames.append(f[:-8])
    mask_files = []
    for f in glob.glob(os.path.join(datapath, "*_mask.tiff")):
        mask_files.append(f)
    print(filenames)
    return filenames

def make_patches(path, patch_size=512, save_path="../miccai_patches/"):
    files =  get_tiff_path(path)
    for file in files:
        wsi_dir = file + "wsi.tiff"
        mask_dir = file + "mask.tiff"
        extract_patches(wsi_dir, patch_size=patch_size, save_path=os.path.join(save_path, f"{file}/patches-{patch_size}/"))
        extract_patches(mask_dir, patch_size=patch_size, save_path=os.path.join(save_path, f"{file}/masks-{patch_size}/"))

    print(f"Done! Totoal {len(files)} file.")

if __name__ == "__main__":
    make_patches(path="/lustre/orion/bif146/world-shared/enzhi/MICCAI", 
                 patch_size=512,
                 save_path="/lustre/orion/bif146/world-shared/enzhi/miccai_patches/")