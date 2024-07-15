import os
import re
import cv2 as cv
import numpy as np
from PIL import Image
import glob 
from pathlib import Path


def patches_merge(slide_dir, patches, resolution):
    pattern = patches
    files = glob.glob(pattern)
    regex = re.compile(r'patches_(.*?)_(.*?).png')
    list_i = []
    list_j = []
    for file in files:
        match = regex.search(file)
        if match:
            list_i.append(match.group(1))
            list_j.append(match.group(2))
    num_patches_width=len(list_i)
    num_patches_height = len(list_j)
    width = resolution*num_patches_width
    height = resolution*num_patches_height
    slide = np.zeros((width, height, 1))
    for i in range(num_patches_width):
        for j in range(num_patches_height):
            # Define the coordinates of the current patch
            left = i * resolution
            upper = j * resolution
            right = min(left + resolution, width)
            lower = min(upper + resolution, height)
            slide[left:right,lower:right] = cv.imread(os.path.join(slide_dir, "patches_{i}_{j}.png"))
    save_path = os.path.join(os.path.dirname(slide_dir), "merged-mask-512.png")
    cv.imwrite(save_path, cv.resize(slide, dsize=(1024,1024)))
    # return slide

def get_patches_path(datapath, patch_type="maskes-512"):
    filenames = os.listdir(os.path.join(datapath,patch_type))
    slides_patches={}
    for f in filenames:
        subdir = os.path.join(datapath, f)
        slides_patches[subdir] = []
        for patches in os.listdir(subdir):
            slides_patches[subdir].append(patches)        
    return slides_patches


def make_slides(path, patch_size=512, resolution=1024, save_path="../miccai_patches/"):
    files =  get_patches_path(path)
    for slide_dir, patches in files.items:
        print(f"Start to merge {slide_dir}.")
        patches_merge(slide_dir, patches=patches, patch_size=patch_size, resolution=resolution)
        break
    print(f"Done! Totoal {len(files)} file.")
    
    
if __name__ == "__main__":
    make_slides(path="/lustre/orion/bif146/world-shared/enzhi/MICCAI", 
                 patch_size=512,
                 save_path="/lustre/orion/bif146/world-shared/enzhi/miccai_patches/")