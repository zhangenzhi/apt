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

def main():
    sample_slice_path =  "/lustre/orion/nro108/world-shared/enzhi/spring8data/8192_output_2/No_020/035"
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
    
    import pdb;pdb.set_trace()
    
    pred_slices = []
    image_slices = []
    for img_name in image_filenames:
        image = np.fromfile(img_name, dtype=np.uint16).reshape([8192, 8192, 1])
        image = (image[:] / 255).astype(np.uint8)
        image = torch.Tensor(image)
        image = (image - image.min()) / (image.max() - image.min()+1e-4)
        image = image.permute(2,0,1).unsqueeze(0)
        save_input_as_image(image[0].permute(1,2,0), f"real_img_{img_name}.png")
        
        with torch.no_grad():
            image = image.to(device=device)
            pred = model(image)
            save_pred_as_mask(pred[0], f"pred_{img_name}.png")
            
            pred = pred[0].argmax(axis=0).permute(1,2,0)
            image = image[0].permute(1,2,0)
            pred_resized = cv2.resize(pred.cpu().numpy(), (512, 512), interpolation=cv2.INTER_NEAREST)
            image_resized = cv2.resize(image.cpu().numpy(), (512, 512), interpolation=cv2.INTER_NEAREST)
            pred_slices.append(pred_resized)
            image_slices.append(image_resized)
    
    pred_slices = np.stack(pred_slices, axis=0)    # (N, 512, 512)
    image_slices = np.stack(image_slices, axis=0) # (N, 512, 512)
    np.savez("output_3d_data.npz", dem=pred_slices, image=image_slices)
    print("Saved as 3D data:", pred_slices.shape, image_slices.shape)

def post_process():
    pass

if __name__ == "__main__":
    main()