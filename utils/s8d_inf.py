import os
import sys
sys.path.append("./")
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
import torch
import tifffile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Set the flag to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    import pdb;pdb.set_trace()
    
    img_name = "/lustre/orion/nro108/world-shared/enzhi/apt/dataset/sample_8192.raw"
    image = np.fromfile(img_name, dtype=np.uint16)
    image = (image[:] / 255).astype(np.uint8)
    image = torch.Tensor(image)
    
if __name__ == "__main__":
    main()