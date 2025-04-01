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
from apt.transforms import ImagePatchify

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
        print("img samples: ", len(self.image_filenames))
        
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

class S8DFinetuneAP(Dataset):
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
        print("img samples: ", len(self.image_filenames))
        
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

class S8DFinetune(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Path to the root directory containing FBPs and labels folders.
            transform (callable, optional): Optional transform to be applied on the FBP images.
            target_transform (callable, optional): Optional transform to be applied on the labels.
        """
        self.root_dir = root_dir
        self.fbp_dir = os.path.join(root_dir, 'FBPs')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.transform = transform
        self.target_transform = target_transform
        
        # Get list of files (without extensions to match FBPs and labels)
        self.fbp_files = [f for f in os.listdir(self.fbp_dir) if f.endswith('.tiff')]
        
        # Verify corresponding labels exist
        self.valid_files = []
        for fbp_file in self.fbp_files:
            # Extract base name (assuming pattern: ..._reconFBPsimul_RingAF_12.tiff)
            base_name = fbp_file.split('_reconFBPsimul_')[0]
            label_file = f"{base_name}_label.tiff"
            if os.path.exists(os.path.join(self.label_dir, label_file)):
                self.valid_files.append((fbp_file, label_file))
            else:
                print(f"Warning: Missing label for {fbp_file}")
        print(self.valid_files)
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        fbp_file, label_file = self.valid_files[idx]
        
        # Load FBP image
        fbp_path = os.path.join(self.fbp_dir, fbp_file)
        fbp_image = Image.open(fbp_path)
        fbp_array = np.array(fbp_image)
        
        # Load label/mask
        label_path = os.path.join(self.label_dir, label_file)
        label_image = Image.open(label_path)
        label_array = np.array(label_image)
        
        # Apply transforms if any
        if self.transform:
            fbp_array = self.transform(fbp_array)
        if self.target_transform:
            label_array = self.target_transform(label_array)
            
        # Convert to tensors
        fbp_tensor = torch.from_numpy(fbp_array).float()
        label_tensor = torch.from_numpy(label_array).long()  # Assuming labels are integers
        
        # Add channel dimension if needed (for 2D images)
        if len(fbp_tensor.shape) == 2:
            fbp_tensor = fbp_tensor.unsqueeze(0)  # Shape: (1, H, W)
        if len(label_tensor.shape) == 2:
            label_tensor = label_tensor.unsqueeze(0)  # Shape: (1, H, W)
            
        return fbp_tensor, label_tensor
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="s8d", 
                        help='base path of dataset.')
    parser.add_argument('--data_dir', default="/lustre/orion/world-shared/lrn075/Riken_XCT_Simulated_Data/8192x8192xN_Simulations/Noise_0.05_Blur_2_sparsity_2_NumAng_3600", 
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
    
    dataset = S8DFinetune(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # # S8DAP  usage
    # dataset = Spring8DatasetAP(args.data_dir, args.resolution)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Now you can iterate over the dataloader to get batches of images and masks
    for batch in dataloader:
        images,_,_ = batch
        print(images.shape)