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
import cv2

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
    
    # dataset = S8DFinetune2DAP(args.data_dir, num_classes=5, fixed_length=4096, patch_size=8)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # # 计算总batch数和要跳过的batch数
    # total_batches = len(dataloader)  # 注意：len(dataloader) = ceil(数据集大小 / batch_size)
    # skip_batches = max(0, total_batches - 140)  # 要跳过的batch数
    # print(f"total_batches:{total_batches}")
    
    # import itertools
    # # 使用 islice 跳过前面的batch，只取最后40个
    # last_40_batches = itertools.islice(dataloader, skip_batches, None)

    # image_list = []
    # mask_list = []
    # dem_list = []
    # # Now you can iterate over the dataloader to get batches of images and masks
    # for batch in last_40_batches:
    #     image, mask, qimages, qmasks, qdt = batch
    #     print(qimages.shape, qmasks.shape)
    #     dem = qdt[0].deserialize(qmasks[0].permute(1,2,0).numpy(), 8, 5)
    #     # dem = np.transpose(dem, (2, 1, 0))
    #     # 3. Resize 到 512x512（使用 OpenCV）
    #     image_np = image[0].numpy()
    #     mask_np = mask[0].numpy()
        
    #     # import pdb;pdb.set_trace()
        
    #     dem_resized = cv2.resize(dem, (512, 512), interpolation=cv2.INTER_NEAREST)
    #     image_resized = cv2.resize(image_np, (512, 512), interpolation=cv2.INTER_NEAREST)
    #     mask_resized = cv2.resize(mask_np, (512, 512), interpolation=cv2.INTER_NEAREST)  # mask 用最近邻
        
    #     dem = torch.from_numpy(dem_resized)
    #     dem = dem.argmax(axis=-1)
    #     image_list.append(image_resized)
    #     mask_list.append(mask_resized)
    #     dem_list.append(dem.numpy())
    
    # # 5. 转为 3D 数据
    # dem_3d = np.stack(dem_list, axis=0)    # (N, 512, 512)
    # image_3d = np.stack(image_list, axis=0) # (N, 512, 512)
    # mask_3d = np.stack(mask_list, axis=0)  # (N, 512, 512)

    # # 6. 保存
    # np.savez("output_3d_data.npz", dem=dem_3d, image=image_3d, mask=mask_3d)
    # print("Saved as 3D data:", dem_3d.shape, image_3d.shape, mask_3d.shape)
    
    # 1. Load the NPZ file
    data = np.load("output_3d_data.npz")

    # 2. Extract arrays
    dem = data["dem"]      # Shape: (N, H, W)
    dem = np.where(dem == 1, dem, 0)  # Replace non-1 values with 0
    image = data["image"]  # Shape: (N, H, W)
    mask = data["mask"]    # Shape: (N, H, W)
    mask = np.where(mask == 1, mask, 0)

    for i in range(40):
        s = dem[i]
        if np.sum(s)>16000:
            dem[i] = dem[i]*0
            mask[i] = mask[i]*0
    dem = dem*(2048)
    mask = mask*(2048)
    
    dem = np.where(dem == 2048, dem, -1024) 
    mask = np.where(mask == 2048, mask, -1024)
    # import pdb;pdb.set_trace()
    dem = dem.astype(np.float32)
    image = image.astype(np.float32)
    mask = mask.astype(np.float32)
    
    # 3. Save each array as raw binary file
    def save_as_raw(array, filename):
        # Flatten the array to 1D and write binary
        with open(filename, "wb") as f:
            array.flatten().tofile(f)

    save_as_raw(dem, "dem.raw")
    save_as_raw(image, "image.raw")
    save_as_raw(mask, "mask.raw")

    print("Saved raw files:")
    print(f"dem.raw   - Shape: {dem.shape}, Dtype: {dem.dtype}")
    print(f"image.raw - Shape: {image.shape}, Dtype: {image.dtype}")
    print(f"mask.raw  - Shape: {mask.shape}, Dtype: {mask.dtype}")