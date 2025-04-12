import os
import sys
sys.path.append("./")
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
import torch
import tifffile
from torch.utils.data import DataLoader

from dataset.s8d_2d import S8DFinetune2DAP, collate_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="s8d", 
                        help='base path of dataset.')
    parser.add_argument('--data_dir', default="/lustre/orion/nro108/world-shared/enzhi/Riken_XCT_Simulated_Data/8192x8192_2d_Simulations/Noise_0.05_Blur_2_sparsity_2_NumAng_3600", 
                        help='base path of dataset.')
    parser.add_argument('--epoch', default=1, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch_size for training')
    args = parser.parse_args()
    
    dataset = S8DFinetune2DAP(args.data_dir, num_classes=5, fixed_length=10201, patch_size=8)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    import pdb;pdb.set_trace()
    
    # Now you can iterate over the dataloader to get batches of images and masks
    for batch in dataloader:
        seq_img, seq_mask, seq_size, seq_pos, qdt = batch
        print(seq_img.shape, seq_mask.shape)