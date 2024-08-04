import os
import sys
sys.path.append("./")
import argparse
from pathlib import Path
import numpy as np
import PIL
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from apt.transforms import Patchify

# Set the flag to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 933120000

class MICCAITrans(Dataset):
    def __init__(self, data_path, resolution, sths=[1,3,5,7], cannys=[50, 100], fixed_length=1024, patch_size=8, eval_mode=False):
        self.data_path = data_path
        self.subslides = os.listdir(data_path) if not eval_mode else ["08-368_01_"]
        self.resolution = resolution

        self.image_filenames = []
        self.mask_filenames = []
        
        data_path_paip = "/lustre/orion/bif146/world-shared/enzhi/paip/output_images_and_masks"
        for subdir in os.listdir(data_path_paip):
            subdir_path = os.path.join(data_path, subdir)
            if os.path.isdir(subdir_path):
                image = os.path.join(subdir_path, f"rescaled_image_0_{resolution}x{resolution}.png")
                mask = os.path.join(subdir_path, f"rescaled_mask_0_{resolution}x{resolution}.png")

                # Ensure the image exist
                if os.path.exists(image) and os.path.exists(mask):
                    self.image_filenames.extend([image])
                    self.mask_filenames.extend([mask])
                    
        print("img tiles: ",len(self.image_filenames))
        for subdir in self.subslides:
            subdir_path = os.path.join(data_path, subdir)
            if os.path.isdir(subdir_path):
                images_path = os.path.join(subdir_path, f"rescale-images-{resolution}/")
                masks_path = os.path.join(subdir_path, f"rescale-masks-{resolution}/")

                for img_name in os.listdir(images_path):
                    # Ensure the image exist
                    image = os.path.join(images_path, img_name)
                    mask = os.path.join(masks_path, img_name)
                    if os.path.exists(image) and os.path.exists(mask):
                        self.image_filenames.extend([image])
                        self.mask_filenames.extend([mask])
                    
        print("img tiles: ",len(self.image_filenames))
        
        self.patchify = Patchify(sths=sths, fixed_length=fixed_length, cannys=cannys, patch_size=patch_size)
        
        self.transform= transforms.Compose([
                transforms.ToTensor(),
            ])
            
        self.transform_mask= transforms.Compose([
            transforms.ToTensor(),
        ])
    
    
    def compute_img_statistics(self):
        # Initialize accumulators for mean and std
        mean_acc = np.zeros(3)
        std_acc = np.zeros(3)

        # Loop through the dataset and accumulate channel-wise mean and std
        for img_name in self.image_filenames:
            img = Image.open(img_name).convert("RGB")
            img_np = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            
            # Accumulate mean and std separately for each channel
            mean_acc += np.mean(img_np, axis=(0, 1))
            std_acc += np.std(img_np, axis=(0, 1))

        # Calculate the overall mean and std
        mean = mean_acc / len(self.image_filenames)
        std = std_acc / len(self.image_filenames)
        print(mean, std)
        return mean.tolist(), std.tolist()
    
    def compute_mask_statistics(self):
        # Initialize accumulators for mean and std
        mean_acc = np.zeros(1)
        std_acc = np.zeros(1)

        # Loop through the dataset and accumulate channel-wise mean and std
        for img_name in self.mask_filenames:
            img = Image.open(img_name).convert("L")
            img_np = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            
            # Accumulate mean and std separately for each channel
            mean_acc += np.mean(img_np, axis=(0, 1))
            std_acc += np.std(img_np, axis=(0, 1))

        # Calculate the overall mean and std
        mean = mean_acc / len(self.mask_filenames)
        std = std_acc / len(self.mask_filenames)

        return mean.tolist(), std.tolist()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")  # Assuming masks are grayscale

        image = np.array(image)
        mask = np.array(mask)
        mask = np.reshape(mask, mask.shape+(1,))
        qdt_img, qdt_mask, qdt = self.patchify(image, mask)
        
        # # Apply transformations
        image = self.transform(image)
        qdt_img = self.transform(qdt_img)
        mask = self.transform(mask)
        qdt_mask = self.transform(qdt_mask)
        
        mask = mask.long()
        mask = F.one_hot(mask, num_classes=2)
        mask = torch.squeeze(mask)
        mask = torch.permute(mask, dims=(2,0,1))
        
        qdt_mask = qdt_mask.long()
        qdt_mask = F.one_hot(qdt_mask, num_classes=2)
        qdt_mask = torch.squeeze(qdt_mask)
        qdt_mask = torch.permute(qdt_mask, dims=(2,0,1))
        qdt_mask = qdt_mask.to(torch.float32)

        qdt_info = qdt.encode_nodes()
        qdt_value = qdt.nodes_value()
        
        return image, qdt_img, mask, qdt_mask, qdt_info, torch.Tensor(qdt_value)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="miccai", 
                        help='base path of dataset.')
    parser.add_argument('--data_dir', default="../miccai_patches/", 
                        help='base path of dataset.')
    parser.add_argument('--resolution', default=512, type=int,
                        help='resolution of img.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./vitunet_visual",
                        help='save visualized and loss filename')
    args = parser.parse_args()

    # Example usage
    dataset = MICCAIDataset(args.data_dir, args.resolution, normalize=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Now you can iterate over the dataloader to get batches of images and masks
    for batch in dataloader:
        images, masks = batch
        print(images.shape, masks.shape)
        # visualize_samples(images, masks, num_samples=4)
        # break
        # Your training/validation loop goes here