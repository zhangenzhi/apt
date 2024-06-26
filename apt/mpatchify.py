import sys
sys.path.append("./")

from apt.quadtree import FixedQuadTree

import cv2 as cv
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from utils.savescsv import savedict, savelist
import argparse

import os 
import glob 
from pathlib import Path

def get_img_path(datapath):
    files = []
    for f in glob.glob(os.path.join(datapath, "*/*.jpeg")):
        files.append(f)
    return files

def get_png_path(base, resolution):
    data_path = base
    image_filenames = []
    mask_filenames = []

    for subdir in os.listdir(data_path):
        subdir_path = os.path.join(data_path, subdir)
        if os.path.isdir(subdir_path):
            image = os.path.join(subdir_path, f"rescaled_image_0_{resolution}x{resolution}.png")
            mask = os.path.join(subdir_path, f"rescaled_mask_0_{resolution}x{resolution}.png")

            # Ensure the image exist
            if os.path.exists(image) and os.path.exists(mask):

                image_filenames.extend([image])
                mask_filenames.extend([mask])
                
    return image_filenames, mask_filenames

def get_imagenet_path(datapath):
    files = []
    for f in glob.glob(os.path.join(datapath, "*/*.jpeg")):
        files.append(f)
    return files
 
def transform(img, sth:int=3, canny:tuple=(100,200), dsize:tuple=(512, 512)):
    res = cv.resize(img, dsize=dsize, interpolation=cv.INTER_CUBIC)
    grey_img = res[:, :, 0]
    blur = cv.GaussianBlur(grey_img, (sth,sth), 0)
    edge = cv.Canny(blur, canny[0], canny[1])
    return res, edge


# save patch sequence
def compress_mix_patches(qdt:FixedQuadTree, img: np.array, to_size:tuple=(8,8,3)):
    seq_patches = qdt.serialize(img, size=to_size)
    return seq_patches, to_size

def paip_patchify(base, fixed_length:int, resolution: int, sth:int=3, to_size:tuple=(8,8,3)):
    img_path, mask_path = get_png_path(base=base, resolution=resolution)
    output_dir = base
    os.makedirs(output_dir, exist_ok=True)
    
    for i,(p,m) in enumerate(zip(img_path, mask_path)):
        img = cv.imread(p)
        mask = cv.imread(m)
        
        mask, edge = transform(mask, sth=sth, dsize=(resolution, resolution))
        qdt = FixedQuadTree(domain=edge, fixed_length=fixed_length)
        seq_patches = qdt.serialize(mask, size=to_size)
        seq_mask = np.asarray(seq_patches)
        seq_mask = np.reshape(seq_mask, [to_size[0], -1, to_size[2]])
        
        seq_img = qdt.serialize(img, size=to_size)
        seq_img = np.asarray(seq_img)
        seq_img = np.reshape(seq_img, [to_size[0], -1, to_size[2]])
        
        name = Path(p).parts[-2]
        cv.imwrite(output_dir+"/{}/mimage-{}_{}_{}_{}.png".format(name, resolution, fixed_length, to_size[0], "qdt"), seq_img)
        cv.imwrite(output_dir+"/{}/mmask-{}_{}_{}_{}.png".format(name, resolution, fixed_length, to_size[0], "qdt"), seq_mask)
                
    print("Fixed lenth:{}, resolution:{}, to_size:{}, sth:{}".format(fixed_length, resolution, to_size[0], sth))
        
def imagenet_patcher(datapath):
    train_path = os.path.join(datapath, "train")
    val_path = os.path.join(datapath, "val")
    save_to =os.path.join(datapath, "imagenet_qdt")
    if not os.path.exists(save_to):
        os.makedirs(save_to)

def patchify(args):
    datapath = args.data_dir
    if args.dataset == "imagenet":
        imagenet_patcher(datapath=datapath)
    elif args.dataset == "paip":
        paip_patchify(base=datapath, 
                      resolution=args.resolution,
                      sth=args.sth,
                      fixed_length=args.fixed_length,
                      to_size=(args.to_size, args.to_size, 3))
    elif args.dataset == "btcv":
        pass
    else:
        pass

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Patchify dataset.')
    parser.add_argument('--dataset', type=str,  default="paip", help='name of the dataset.')
    parser.add_argument('--resolution', type=int, default=512, help='resolution of the img.')
    parser.add_argument('--to_size', type=int, default=8, help='path of the dataset.')
    parser.add_argument('--fixed_length', type=int, default=576, help='path of the dataset.')
    parser.add_argument('--sth', type=int, default=3, help='smooth factor for gaussain smoothing.')
    parser.add_argument('--data_dir',  type=str, default="/Volumes/data/dataset/paip/output_images_and_masks", 
                        help='base path of dataset.')
    args = parser.parse_args()
    
    import time
    start_time = time.time()
    patchify(args)
    print("patchify cost time {}".format(time.time()-start_time))