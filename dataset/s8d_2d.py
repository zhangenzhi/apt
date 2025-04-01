import os
import sys
sys.path.append("./")
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from map.transform import ImagePatchify

# Set the flag to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Spring8Dataset(Dataset):
    def __init__(self, data_path, resolution):
        self.data_path = data_path
        self.resolution = resolution
        self.subslides = os.listdir(data_path)
        self.image_filenames = []

        for subdir in self.subslides:
            subdir_path = os.path.join(data_path, subdir)
            if os.path.isdir(subdir_path):
                sample_path = os.listdir(subdir_path)
                for sampledir in sample_path:
                    sample_slice_path = os.path.join(subdir_path, sampledir)
                    if os.path.isdir(sample_slice_path):
                        num_sample_slice = len(os.listdir(sample_slice_path))
                        for i in range(num_sample_slice):
                            # Ensure the image exist
                            img_name = f"volume_{str(i).zfill(3)}.raw"
                            image = os.path.join(sample_slice_path, img_name)
                            if os.path.exists(image):
                                self.image_filenames.extend([image])
        print("img tiles: ", len(self.image_filenames))
        
        self.transform= transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image = np.fromfile(img_name, dtype=np.uint16).reshape([self.resolution, self.resolution, 1])
        image = (image[:] / 255).astype(np.uint8)
        image = self.transform(image)
        return image

class Spring8DatasetAP(Dataset):
    def __init__(self, data_path, resolution, fixed_length=1024, sths=[0,1,3,5,7], cannys=[50, 100], patch_size=16, ):
        self.data_path = data_path
        self.resolution = resolution
        self.patchify = ImagePatchify(sths=sths, fixed_length=fixed_length, cannys=cannys, patch_size=patch_size, num_channels=1)

        self.subslides = os.listdir(data_path)
        self.image_filenames = []

        for subdir in self.subslides:
            subdir_path = os.path.join(data_path, subdir)
            if os.path.isdir(subdir_path):
                sample_path = os.listdir(subdir_path)
                for sampledir in sample_path:
                    sample_slice_path = os.path.join(subdir_path, sampledir)
                    if os.path.isdir(sample_slice_path):
                        num_sample_slice = len(os.listdir(sample_slice_path))
                        for i in range(num_sample_slice):
                            # Ensure the image exist
                            img_name = f"volume_{str(i).zfill(3)}.raw"
                            image = os.path.join(sample_slice_path, img_name)
                            if os.path.exists(image):
                                self.image_filenames.extend([image])
        print("img tiles: ", len(self.image_filenames))
        
        self.transform= transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image = np.fromfile(img_name, dtype=np.uint16).reshape([self.resolution, self.resolution, 1])
        image = (image[:] / 255).astype(np.uint8)
        seq_img, seq_size, seq_pos = self.patchify(image)
        seq_img = self.transform(seq_img)
        return seq_img, seq_size, seq_pos
    
class S8DGanAP(Dataset):
    def __init__(self, data_path, resolution, fixed_length=1024, sths=[0,1,3,5,7], cannys=[50, 100], patch_size=16, ):
        self.data_path = data_path
        self.resolution = resolution
        self.patchify = ImagePatchify(sths=sths, fixed_length=fixed_length, cannys=cannys, patch_size=patch_size, num_channels=1)

        self.subslides = os.listdir(data_path)
        self.image_filenames = []

        for subdir in self.subslides:
            subdir_path = os.path.join(data_path, subdir)
            if os.path.isdir(subdir_path):
                sample_path = os.listdir(subdir_path)
                for sampledir in sample_path:
                    sample_slice_path = os.path.join(subdir_path, sampledir)
                    if os.path.isdir(sample_slice_path):
                        num_sample_slice = len(os.listdir(sample_slice_path))
                        for i in range(num_sample_slice):
                            # Ensure the image exist
                            img_name = f"volume_{str(i).zfill(3)}.raw"
                            image = os.path.join(sample_slice_path, img_name)
                            if os.path.exists(image):
                                self.image_filenames.extend([image])
        print("img tiles: ", len(self.image_filenames))
        
        self.transform= transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image = np.fromfile(img_name, dtype=np.uint16).reshape([self.resolution, self.resolution, 1])
        image = (image[:] / 255).astype(np.uint8)
        seq_img, seq_size, seq_pos = self.patchify(image)
        seq_img = self.transform(seq_img)
        return seq_img, seq_size, seq_pos
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="miccai", 
                        help='base path of dataset.')
    parser.add_argument('--data_dir', default="/lustre/orion/nro108/world-shared/enzhi/spring8data/demo", 
                        help='base path of dataset.')
    parser.add_argument('--resolution', default=8192, type=int,
                        help='resolution of img.')
    parser.add_argument('--epoch', default=1, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch_size for training')
    args = parser.parse_args()

    # # S8D  usage
    # dataset = Spring8Dataset(args.data_dir, args.resolution)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    dataset = S8DGanAP(args.data_dir, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # # S8DAP  usage
    # dataset = Spring8DatasetAP(args.data_dir, args.resolution)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Now you can iterate over the dataloader to get batches of images and masks
    for batch in dataloader:
        images,_,_ = batch
        print(images.shape)