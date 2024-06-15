import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    ResizeWithPadOrCrop,
    Resized,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

import torch

# print_config()

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # Resized(keys=["image","label"],spatial_size=[96,96,96]),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)

def btcv(args):
    data_dir = args.data_dir
    split_json = "dataset.json"

    datasets = os.path.join(data_dir,split_json)
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_ds = CacheDataset(data=val_files, 
                          transform=val_transforms, 
                          cache_num=6, 
                          cache_rate=1.0, 
                          num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    dataloaders = {'train': train_loader,  'val': val_loader}
    datasets = {'train': train_ds,  'val': val_ds}
    return dataloaders, datasets

def visualize(val_ds):
    slice_map = {
        "img0035.nii.gz": 170,
        "img0036.nii.gz": 230,
        "img0037.nii.gz": 204,
        "img0038.nii.gz": 204,
        "img0039.nii.gz": 204,
        "img0040.nii.gz": 180,
    }

    for case_num in range(len(val_ds)):
        img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        img_shape = img.shape
        label_shape = label.shape
        print(f"image shape: {img_shape}, label shape: {label_shape}")
        plt.figure("image", (18, 6))
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
        plt.savefig("btcv_{}.png".format(case_num))
    
def btcv_iter(args):

    dataloaders = btcv(args=args)
    
    # Example usage:
    # Iterate through the dataloaders
    import time
    for e in range(args.num_epochs):
        start_time = time.time()
        for phase in ['train', 'val']:
            for step, sample in enumerate(dataloaders[phase]):
                if step%6==0:
                    print(step, sample["image"].shape, sample["label"].shape)
        print("Time cost for loading {}".format(time.time() - start_time))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="btcv", 
                        help='base path of dataset.')
    parser.add_argument('--data_dir', default="/Volumes/data/dataset/btcv/data", 
                        help='base path of dataset.')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch_size for training')
    args = parser.parse_args()
   
    btcv_iter(args=args)