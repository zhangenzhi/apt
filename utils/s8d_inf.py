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

# Set the flag to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    import pdb;pdb.set_trace()
    
    img_name = "/lustre/orion/nro108/world-shared/enzhi/apt/dataset/sample_8192.raw"
    image = np.fromfile(img_name, dtype=np.uint16).reshape([8192, 8192, 1])
    image = (image[:] / 255).astype(np.uint8)
    image = torch.Tensor(image)
    save_input_as_image(image, "real_img.png")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    num_class = 5
    model = Unet(n_class=num_class, in_channels=1, pretrained=False)
    model = model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join("/lustre/orion/nro108/world-shared/enzhi/apt/unet-s8d-n32-dce", "best_score_model.pth")))
    with torch.no_grad():
        pred = model(image)
        save_pred_as_mask(pred, "pred.png")
        
if __name__ == "__main__":
    main()