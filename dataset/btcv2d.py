import os
import logging
import sys
sys.path.append("./")
import torch
from torchvision.utils import save_image

from dataset.btcv import btcv

import logging
logging.disable(logging.WARNING)
def log(args):
    logging.basicConfig(
        filename=os.path.join(args.output, args.logname),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def btcv2d(args):
    os.makedirs(args.output, exist_ok=True)
    log(args=args)
    # Create DataLoader for training and validation
    dataloaders, datasets = btcv(args=args)
    
    for i, sample in enumerate(dataloaders['train']):
        img_name = os.path.split(sample["image"].meta["filename_or_obj"][0])[1]
        images = sample["image"]
        labels = sample["label"]
        
        s0 = images[0]
        s0 = torch.permute(s0, (1, 0, 2, 3))
        for i in range(s0.shape[0]):
            path = os.path.join(args.output,'image-{}-{}.png'.format(img_name, i))
            save_image(s0[i], path)
            
        t0 = labels[0]
        t0 = torch.permute(t0, (1, 0, 2, 3))
        for i in range(t0.shape[0]):
            path = os.path.join(args.output,'mask-{}-{}.png'.format(img_name, i))
            save_image(t0[i], path)
            
    for i, batch in enumerate(dataloaders['val']):            
        img_name = os.path.split(batch["image"].meta["filename_or_obj"][0])[1]
        images = batch["image"][0]
        labels = batch["label"][0]
        
        s0 = images[0]
        s0 = torch.permute(s0, (1, 0, 2, 3))
        for i in range(s0.shape[0]):
            path = os.path.join(args.output,'image-{}-{}.png'.format(img_name, i))
            save_image(s0[i], path)
            
        t0 = labels[0]
        t0 = torch.permute(t0, (1, 0, 2, 3))
        for i in range(t0.shape[0]):
            path = os.path.join(args.output,'mask-{}-{}.png'.format(img_name, i))
            save_image(t0[i], path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='BTCV 3D to 2D and loader')
    parser.add_argument('--logname', type=str, default='btcv2d.log', help='logging of task.')
    parser.add_argument('--output', type=str, default='./output', help='output dir')
    parser.add_argument('--data_dir', type=str, default='/Volumes/data/dataset/btcv/data', help='Path to the BTCV dataset directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    args = parser.parse_args()
    btcv2d(args)