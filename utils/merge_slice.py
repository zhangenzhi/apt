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
    
    dataset = S8DFinetune2DAP(args.data_dir, num_classes=5, fixed_length=4096, patch_size=8)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 计算总batch数和要跳过的batch数
    total_batches = len(dataloader)  # 注意：len(dataloader) = ceil(数据集大小 / batch_size)
    skip_batches = max(0, total_batches - 40)  # 要跳过的batch数
    print(f"total_batches:{total_batches}")
    
    import itertools
    # 使用 islice 跳过前面的batch，只取最后40个
    last_40_batches = itertools.islice(dataloader, skip_batches, None)

    image_list = []
    mask_list = []
    dem_list = []
    # Now you can iterate over the dataloader to get batches of images and masks
    for batch in last_40_batches:
        image, mask, qimages, qmasks, qdt = batch
        print(qimages.shape, qmasks.shape)
        dem,_,_ = qdt[0].deserialize(qmasks.permute(1,2,0).numpy(), 8, 5)
        dem = np.transpose(dem, (2, 1, 0))
        image_list.append(image)
        mask_list.append(mask)
        dem_list.append(dem)
    
    # 5. 合并所有 batch 的数据
    dem_stack = torch.cat(dem_list, dim=0)    # (N, 512, 512)
    image_stack = torch.cat(image_list, dim=0) # (N, 512, 512)
    mask_stack = torch.cat(mask_list, dim=0)  # (N, 512, 512)

    # 6. 保存为 3D 数据（npz 格式）
    np.savez("output_3d_data.npz", 
            dem=dem_stack.numpy(),
            image=image_stack.numpy(),
            mask=mask_stack.numpy())

    print("Saved as 3D data:", dem_stack.shape, image_stack.shape, mask_stack.shape)