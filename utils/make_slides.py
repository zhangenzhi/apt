import os
import re
import cv2 as cv
import numpy as np
from PIL import Image
import glob 
from pathlib import Path


def patches_merge(slide_dir, patches, patch_size, patch_type, resolution):
    patch_type_dir = os.path.join(slide_dir, patch_type)
    files = patches
    regex = re.compile(r'patch_(.*?)_(.*?).png')
    list_i = []
    list_j = []
    for file in files:
        match = regex.search(file)
        if match:
            list_i.append(match.group(1))
            list_j.append(match.group(2))
            
    num_patches_width = int(max([int(idx) for idx in list_i]))
    num_patches_height = int(max([int(idx) for idx in list_j]))
    width = patch_size*(num_patches_width+1)
    height = patch_size*(num_patches_height+1)
    print(width, height)
    # import pdb
    # pdb.set_trace()
    slide = np.zeros((width, height))
    for i in range(num_patches_width+1):
        for j in range(num_patches_height+1):
            # Define the coordinates of the current patch
            left = i * patch_size
            right = min(left + patch_size, width)
            lower = j * patch_size
            upper = min(lower + patch_size, height)
            slide[left:right,lower:upper] = cv.imread(os.path.join(patch_type_dir, f"patch_{i}_{j}.png"), cv.IMREAD_GRAYSCALE)
    save_path = os.path.join(os.path.dirname(patch_type_dir), "merged-predict-mask-512.png")
    cv.imwrite(save_path, cv.resize(slide, dsize=(resolution, resolution)))

def get_patches_path(datapath, patch_type="masks-512"):
    # filenames = os.listdir(datapath)
    filenames=["08-368_01_"]
    slides_patches={}
    for f in filenames:
        subdir = os.path.join(datapath, f)
        slides_patches[subdir] = []
        for patches in os.listdir(os.path.join(subdir, patch_type)):
            slides_patches[subdir].append(patches)        
    return slides_patches


def make_slides(path, patch_size=512, resolution=1024, patch_type="masks-512", save_path="../miccai_patches/"):
    files =  get_patches_path(path, patch_type=patch_type)
    for slide_dir, patches in files.items():
        print(f"Start to merge {slide_dir}.")
        patches_merge(slide_dir, patches=patches, patch_size=patch_size, patch_type=patch_type, resolution=resolution)
        break
    print(f"Done! Totoal {len(files)} file.")
    
    
if __name__ == "__main__":
    make_slides(path="/lustre/orion/bif146/world-shared/enzhi/miccai_patches", 
                 patch_size=512,
                 patch_type="predict_patches-512",
                 save_path="/lustre/orion/bif146/world-shared/enzhi/miccai_patches/")