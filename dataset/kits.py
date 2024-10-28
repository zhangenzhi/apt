import os
import math
import random
import shutil
import tempfile
import nibabel as nib
from torch.utils.data.dataset import Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class KITS(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        self.image_filenames, self.mask_filenames = self.get_kits_files(datapath)
        
        self.transform= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([512,512]),
        ])
        
        self.transform_mask= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([512,512]),
        ])
        
    def get_kits_files(self, datapath):
        data_type = ["data", "data_downsampled", "data_permuted", "images"]
        path = os.path.join(datapath, data_type[0])
        filepath = {"image":[], "label":[]}
        for i in range(300):
            case_name = f"case_{str(i).zfill(5)}"
            if case_name in os.listdir(path):
                case_path = os.path.join(path, case_name)
                sample_name = "imaging.nii.gz"
                mask_name = "segmentation.nii.gz"
                # print(case_path)
                if sample_name and mask_name in os.listdir(case_path):
                    filepath["image"] += [os.path.join(case_path, sample_name)]
                    filepath["label"] += [os.path.join(case_path, mask_name)]
                else:
                    continue
                    # print(case_path)
        image_filenames = filepath["image"]
        mask_filenames = filepath["label"]
        return image_filenames, mask_filenames
    
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        img = nib.load(img_name).get_fdata()
        # Convert to a PyTorch tensor
        img = torch.from_numpy(img).float()

        mask = nib.load(mask_name).get_fdata()
        # Convert to a PyTorch tensor
        mask = torch.from_numpy(mask).float()
        
        # Apply any transformations, if specified
        # img = self.transform(img)
        # mask = self.transform_mask(mask)
        
        return img, mask, self.image_filenames[idx]
    
from torchvision import transforms
class KITS2D(Dataset):
    def __init__(self, datapath,  mode="train"):
        self.datapath = datapath
        if mode == "train":
            self.image_filenames, self.mask_filenames = self.get_kits_files(datapath)
        else:
            self.image_filenames, self.mask_filenames = self.get_random_sample(datapath)
            
        self.transform= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256]),
        ])
        # Define the transformations
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),      # Random horizontal flip
            transforms.RandomVerticalFlip(),        # Random vertical flip
            transforms.RandomRotation(15),          # Random rotation between -15 and 15 degrees
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation by 10%
            transforms.ColorJitter(brightness=0.1, contrast=0.1),      # Random brightness/contrast
            transforms.GaussianBlur(kernel_size=3), # Apply Gaussian blur
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.9, 1.1)), # Random scaling
        ])
        
        self.transform_mask= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256]),
        ])
    def get_random_sample(self, datapath, data_type="2d-slices-filt"):
        path = os.path.join(datapath, data_type)
        filepath = {"image":[], "label":[]}
        while len(filepath["image"])==0 and len(filepath["label"])==0:
            i = random.randint(0, len(os.listdir(path)))
            case_name = f"case_{str(i).zfill(5)}"
            if case_name in os.listdir(path):
                case_path = os.path.join(path, case_name)
                for j in range(len(os.listdir(case_path))):
                    sample_name = f"case_{str(i).zfill(5)}_image_slice_{str(j).zfill(5)}.png"
                    mask_name = f"case_{str(i).zfill(5)}_mask_slice_{str(j).zfill(5)}.png"
                    # print(case_path)
                    if sample_name and mask_name in os.listdir(case_path):
                        filepath["image"] += [os.path.join(case_path, sample_name)]
                        filepath["label"] += [os.path.join(case_path, mask_name)]                 
        image_filenames = filepath["image"]
        mask_filenames = filepath["label"]
        return image_filenames, mask_filenames
    
    def get_kits_files(self, datapath, data_type="2d-slices-filt"):
        path = os.path.join(datapath, data_type)
        filepath = {"image":[], "label":[]}
        for i in range(len(os.listdir(path))):
            case_name = f"case_{str(i).zfill(5)}"
            if case_name in os.listdir(path):
                case_path = os.path.join(path, case_name)
                for j in range(len(os.listdir(case_path))):
                    sample_name = f"case_{str(i).zfill(5)}_image_slice_{str(j).zfill(5)}.png"
                    mask_name = f"case_{str(i).zfill(5)}_mask_slice_{str(j).zfill(5)}.png"
                    # print(case_path)
                    if sample_name and mask_name in os.listdir(case_path):
                        filepath["image"] += [os.path.join(case_path, sample_name)]
                        filepath["label"] += [os.path.join(case_path, mask_name)]                 
        image_filenames = filepath["image"]
        mask_filenames = filepath["label"]
        return image_filenames, mask_filenames
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        image = Image.open(img_name).convert("L")
        mask = Image.open(mask_name).convert("L")  # Assuming masks are grayscale

        image = np.array(image)
        mask = np.array(mask)
        
        # # Apply transformations
        image = self.transform(image)
        mask = self.transform_mask(mask)
        
        return image, mask
    
class KITS2DV(Dataset):
    def __init__(self, datapath,  mode="train"):
        self.datapath = datapath
        if mode == "train":
            self.image_filenames, self.mask_filenames, self.sample_flag = self.get_kits_files(datapath)
        else:
            self.image_filenames, self.mask_filenames = self.get_random_sample(datapath)
            
        self.transform= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256]),
        ])
        
        self.transform_mask= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256]),
        ])
    def get_random_sample(self, datapath, data_type="2d-slices-filt"):
        path = os.path.join(datapath, data_type)
        filepath = {"image":[], "label":[]}
        while len(filepath["image"])==0 and len(filepath["label"])==0:
            i = random.randint(0, len(os.listdir(path)))
            case_name = f"case_{str(i).zfill(5)}"
            if case_name in os.listdir(path):
                case_path = os.path.join(path, case_name)
                for j in range(len(os.listdir(case_path))):
                    sample_name = f"case_{str(i).zfill(5)}_image_slice_{str(j).zfill(5)}.png"
                    mask_name = f"case_{str(i).zfill(5)}_mask_slice_{str(j).zfill(5)}.png"
                    # print(case_path)
                    if sample_name and mask_name in os.listdir(case_path):
                        filepath["image"] += [os.path.join(case_path, sample_name)]
                        filepath["label"] += [os.path.join(case_path, mask_name)]                 
        image_filenames = filepath["image"]
        mask_filenames = filepath["label"]
        return image_filenames, mask_filenames
    
    def get_kits_files(self, datapath, data_type="2d-slices-filt"):
        path = os.path.join(datapath, data_type)
        filepath = {"image":[], "label":[], "sample_flag":[]}
        for i in range(len(os.listdir(path))):
            case_name = f"case_{str(i).zfill(5)}"
            if case_name in os.listdir(path):
                case_path = os.path.join(path, case_name)
                flag = 0
                for j in range(len(os.listdir(case_path))):
                    sample_name = f"case_{str(i).zfill(5)}_image_slice_{str(j).zfill(5)}.png"
                    mask_name = f"case_{str(i).zfill(5)}_mask_slice_{str(j).zfill(5)}.png"
                    # print(case_path)
                    if sample_name and mask_name in os.listdir(case_path):
                        filepath["image"] += [os.path.join(case_path, sample_name)]
                        filepath["label"] += [os.path.join(case_path, mask_name)]
                        if flag == 0:
                            filepath["sample_flag"] += [0]
                            flag = 1
                        else:
                            filepath["sample_flag"] += [j]

                                      
        image_filenames = filepath["image"]
        mask_filenames = filepath["label"]
        return image_filenames, mask_filenames, filepath["sample_flag"]
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]
        sample_flag = self.sample_flag[idx]

        image = Image.open(img_name).convert("L")
        mask = Image.open(mask_name).convert("L")  # Assuming masks are grayscale

        image = np.array(image)
        mask = np.array(mask)
        
        # # Apply transformations
        image = self.transform(image)
        mask = self.transform_mask(mask)
        
        return image, mask, sample_flag

class KITS2DVBT(Dataset):
    def __init__(self, datapath,  mode="train"):
        self.datapath = datapath
        if mode == "train":
            self.image_filenames, self.mask_filenames, self.sample_flag = self.get_kits_files(datapath)
        else:
            self.image_filenames, self.mask_filenames = self.get_random_sample(datapath)
        
        # import pdb
        # pdb.set_trace()
        
        self.transform= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256]),
        ])
        
        self.transform_mask= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256]),
        ])
    def get_random_sample(self, datapath, data_type="2d-slices-filt"):
        path = os.path.join(datapath, data_type)
        filepath = {"image":[], "label":[], "bt_gt":[]}
        while len(filepath["image"])==0 and len(filepath["label"])==0:
            i = random.randint(0, len(os.listdir(path)))
            case_name = f"case_{str(i).zfill(5)}"
            if case_name in os.listdir(path):
                case_path = os.path.join(path, case_name)
                for j in range(len(os.listdir(case_path))):
                    sample_name = f"case_{str(i).zfill(5)}_image_slice_{str(j).zfill(5)}.png"
                    mask_name = f"case_{str(i).zfill(5)}_mask_slice_{str(j).zfill(5)}.png"
                    # print(case_path)
                    if sample_name and mask_name in os.listdir(case_path):
                        filepath["image"] += [os.path.join(case_path, sample_name)]
                        filepath["label"] += [os.path.join(case_path, mask_name)]                  
        image_filenames = filepath["image"]
        mask_filenames = filepath["label"]
        return image_filenames, mask_filenames
    
    def get_kits_files(self, datapath, data_type="2d-slices-filt"):
        path = os.path.join(datapath, data_type)
        filepath = {"image":[], "label":[], "sample_flag":[], "bt_gt":[]}
        for i in range(len(os.listdir(path))):
            case_name = f"case_{str(i).zfill(5)}"
            if case_name in os.listdir(path):
                case_path = os.path.join(path, case_name)
                flag = 0
                for j in range(len(os.listdir(case_path))):
                    sample_name = f"case_{str(i).zfill(5)}_image_slice_{str(j).zfill(5)}.png"
                    mask_name = f"case_{str(i).zfill(5)}_mask_slice_{str(j).zfill(5)}.png"
                    # print(case_path)
                    if sample_name and mask_name in os.listdir(case_path):
                        filepath["image"] += [os.path.join(case_path, sample_name)]
                        filepath["label"] += [os.path.join(case_path, mask_name)]
                        if flag == 0:
                            filepath["sample_flag"] += [0]
                            flag = 1
                            filepath["bt_gt"] += ["first_mem"]
                            bt_gt = os.path.join(case_path, mask_name)
                        else:
                            filepath["sample_flag"] += [j]
                            filepath["bt_gt"] += [bt_gt]
                            bt_gt = os.path.join(case_path, mask_name)

                                      
        image_filenames = filepath["image"]
        mask_filenames = filepath["label"]
        return image_filenames, mask_filenames, filepath["sample_flag"]
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]
        sample_flag = self.sample_flag[idx]
        
        # # bootstrap with closest mask
        # if idx == 0:
        #     mem = np.zeros(shape=(256,256,1),dtype=np.float32)
        # else:
        #     mem_name = self.mask_filenames[idx-1]
        #     mem = Image.open(mem_name).convert("L")
        
        # bootstrap with closest mask
        if idx == 0:
            mem = np.zeros(shape=(256,256,1),dtype=np.float32)
        else:
            mem = np.zeros(shape=(256,256,1),dtype=np.float32)
            # shift = int(np.random.randint(low=0, high=idx))
            # mem_name = self.mask_filenames[idx-shift]
            # mem = Image.open(mem_name).convert("L")

        image = Image.open(img_name).convert("L")
        mask = Image.open(mask_name).convert("L")  # Assuming masks are grayscale
       
        image = np.array(image)
        mask = np.array(mask)
        
        # # Apply transformations
        image = self.transform(image)
        mask = self.transform_mask(mask)
        mem = self.transform_mask(mem)
        
        return image, mask, sample_flag, mem
    
def kits2d(args):
    dataset = KITS2D(args.data_dir)

    dataset_size = len(dataset)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size
    test_size = val_size

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, dataset_size))
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = val_loader
    print("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, test_size))
    
    dataloaders = {'train': train_loader,  'val': val_loader}
    # datasets = {'train': train_ds,  'val': val_ds}
    return dataloaders

def kits(args):
    dataset = KITS(args.data_dir)

    dataset_size = len(dataset)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size
    test_size = val_size

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, dataset_size))
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = val_loader
    print("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, test_size))
    
    dataloaders = {'train': train_loader,  'val': val_loader}
    # datasets = {'train': train_ds,  'val': val_ds}
    return dataloaders

def kits_iter(args):

    dataloaders = kits(args=args)
    
    # Example usage:
    # Iterate through the dataloaders
    import time
    for e in range(args.num_epochs):
        start_time = time.time()
        for phase in ['train', 'val']:
            for step, sample in enumerate(dataloaders[phase]):
                if step%6==0:
                    print(step, sample[0].shape, sample[1].shape)
        print("Time cost for loading {}".format(time.time() - start_time))
        
def kits2d_iter(args):

    dataloaders = kits2d(args=args)
    
    # Example usage:
    # Iterate through the dataloaders
    import time
    for e in range(args.num_epochs):
        start_time = time.time()
        for phase in ['train', 'val']:
            for step, sample in enumerate(dataloaders[phase]):
                if step%6==0:
                    print(step, sample[0].shape, sample[1].shape)
        print("Time cost for loading {}".format(time.time() - start_time))
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="kits", 
                        help='base path of dataset.')
    # parser.add_argument('--data_dir', default="/mnt/nfs-mnj-archive-12/group/ext-medimg-pe/datasets/public/kits19", 
    #                     help='base path of dataset.')
    parser.add_argument('--data_dir', default="/mnt/nfs-mnj-home-43/i24_enzhang/kits19", 
                        help='base path of dataset.')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch_size for training')
    args = parser.parse_args()
   
    # kits_iter(args=args)
    kits2d_iter(args=args)