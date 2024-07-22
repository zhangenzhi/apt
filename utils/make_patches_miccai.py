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

def rescale_slides(image_path, target_size=16384, save_path='rescale-images/'):
    # Open the TIFF image using OpenSlide
    slide = openslide.OpenSlide(image_path)
    
    # Ensure that the save_path directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Calculate the dimensions at the specified zoom level
    scaled_width = slide.level_dimensions[0][0]
    scaled_height = slide.level_dimensions[0][1]

    # Read the region at the specified zoom level
    scaled_image = slide.read_region((0, 0), 0, (scaled_width, scaled_height))
    scaled_image = scaled_image.resize((target_size,target_size), Image.Resampling.LANCZOS)

    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the rescaled images
    output_image_path = os.path.join(save_path, f"rescaled_image_0_{target_size}x{target_size}.png")
    scaled_image.save(output_image_path)
    print(output_image_path, ", Done!")
    # Close the OpenSlide image
    slide.close()

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

def make_patches(path, patch_size=512, save_path="../miccai_patches/", target_size=16384, task="patches"):
    files =  get_tiff_path(path)
    for file in files:
        wsi_dir = file + "wsi.tiff"
        mask_dir = file + "mask.tiff"
        if task=="patches":
            wsi_save_path = os.path.join(save_path, f"{os.path.basename(file)}/patches-{patch_size}")
            mask_save_path = os.path.join(save_path, f"{os.path.basename(file)}/masks-{patch_size}")
            extract_patches(wsi_dir, patch_size=patch_size, save_path=wsi_save_path)
            extract_patches(mask_dir, patch_size=patch_size, save_path=mask_save_path)
        elif task=="rescale":
            wsi_save_path = os.path.join(save_path, f"{os.path.basename(file)}/rescale-images-{target_size}")
            mask_save_path = os.path.join(save_path, f"{os.path.basename(file)}/rescale-masks-{target_size}")
            rescale_slides(wsi_dir, target_size=target_size, save_path=wsi_save_path)
            rescale_slides(mask_dir, target_size=target_size, save_path=mask_save_path)

    print(f"Done! Totoal {len(files)} file.")

if __name__ == "__main__":
    make_patches(path="/lustre/orion/bif146/world-shared/enzhi/MICCAI", 
                # patch_size=8192,
                # task="patches",
                target_size=16384,
                task="rescale",
                save_path="/lustre/orion/bif146/world-shared/enzhi/miccai_patches/")