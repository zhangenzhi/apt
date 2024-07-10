import os
import sys
sys.path.append("./")
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.apt import APT
from model.sam import SAMQDT
from model.unet import Unet
from dataset.miccai import MICCAIDataset
# from dataset.paip_mqdt import PAIPQDTDataset

import logging

# Configure logging
def log(args):
    os.makedirs(args.savefile, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.savefile, "out.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
# Define the Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        predicted = torch.sigmoid(predicted)
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target) + self.smooth
        dice_coefficient = (2 * intersection + self.smooth) / union
        loss = 1.0 - dice_coefficient  # Adjusted to ensure non-negative loss
        return loss
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.5, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        coeff = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)                                        
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = self.weight*BCE + (1-self.weight)*dice_loss
        
        return Dice_BCE, coeff

    
def main(args, device_id):
    
    # Create an instance of the U-Net model and other necessary components
    model = Unet(n_class=1)
    criterion = DiceBCELoss()
    
    # Move the model to GPU
    model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Define the learning rate scheduler
    # milestones =[int(epoch*r) for r in [0.5, 0.75, 0.875]]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Split the dataset into train, validation, and test sets
    data_path = args.data_dir
    dataset = MICCAIDataset(data_path, args.resolution, normalize=False)
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = (dataset_size - train_size) // 2
    test_size = dataset_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,  sampler=val_sampler)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,  sampler=test_sampler)

    # Training loop
    num_epochs = args.epoch
    train_losses = []
    val_losses = []
    output_dir = args.savefile  # Change this to the desired directory
    os.makedirs(output_dir, exist_ok=True)
    import time
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        
        start_time = time.time()
        step=1
        for batch in train_loader:
            qimages, qmasks = batch
            qimages, qmasks = qimages.to(device_id), qmasks.to(device_id)  # Move data to GPU
            
            outputs = model(qimages)
            loss,_ = criterion(outputs, qmasks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print("train step loss:{}, sec/step:{}".format(loss, (time.time()-start_time)/step))
            epoch_train_loss += loss.item()
            step+=1
        end_time = time.time()
        if device_id == 0:
            logging.info("epoch cost:{}, sec/img:{}".format(end_time-start_time, (end_time-start_time)/train_size))

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        # scheduler.step()

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_score = 0.0

        if device_id == 0:
            with torch.no_grad():
                for batch in val_loader:
                    qimages, qmasks = batch
                    qimages, qmasks = qimages.to(device_id), qmasks.to(device_id)  # Move data to GPU
                    outputs = model(qimages)
                    loss, score = criterion(outputs, qmasks)
                    epoch_val_loss += loss.item()
                    epoch_val_score += score.item()

        epoch_val_loss /= len(val_loader)
        epoch_val_score /= len(val_loader)
        val_losses.append(epoch_val_loss)

        if device_id == 0:
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Score: {epoch_val_score:.4f}.")

        # # Visualize and save predictions on a few validation samples
        # if (epoch + 1) % 3 == 0 and device_id == 0:  # Adjust the frequency of visualization
        #     model.eval()
        #     with torch.no_grad():
        #         qsample_images, qsample_masks= next(iter(val_loader))
        #         qsample_images, qsample_masks = qsample_images.to(device), qsample_masks.to(device)  # Move data to GPU
        #         qsample_outputs = torch.sigmoid(model(qsample_images))

        #         for i in range(qsample_images.size(0)):
        #             image = qsample_images[i].cpu().permute(1, 2, 0).numpy()
        #             mask_true = qsample_masks[i].cpu().numpy()
        #             mask_pred = (qsample_outputs[i, 0].cpu() > 0.5).numpy()
                    
        #             # Squeeze the singleton dimension from mask_true
        #             mask_true = np.squeeze(mask_true, axis=0)

        #             # # Plot and save images
        #             # plt.figure(figsize=(12, 4))
        #             # plt.subplot(1, 3, 1)
        #             # plt.imshow(image)
        #             # plt.title("Input Image")

        #             # plt.subplot(1, 3, 2)
        #             # plt.imshow(mask_true, cmap='gray')
        #             # plt.title("True Mask")

        #             # plt.subplot(1, 3, 3)
        #             # plt.imshow(mask_pred, cmap='gray')
        #             # plt.title("Predicted Mask")

        #             # plt.savefig(os.path.join(output_dir, f"epoch_{epoch + 1}_sample_{i + 1}.png"))
        #             # plt.close()

    # Save train and validation losses
    train_losses_path = os.path.join(output_dir, 'train_losses.pth')
    val_losses_path = os.path.join(output_dir, 'val_losses.pth')
    torch.save(train_losses, train_losses_path)
    torch.save(val_losses, val_losses_path)

    # Test the model
    if device_id == 0:
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                qimages, qmasks = batch
                qimages, qmasks = qimages.to(device_id), qmasks.to(device_id)  # Move data to GPU
                outputs = model(qimages)
                loss,_ = criterion(outputs, qmasks)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        logging.info(f"Test Loss: {test_loss:.4f}")
        draw_loss(output_dir=output_dir)

def draw_loss(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load saved losses
    train_losses_path = os.path.join(output_dir, 'train_losses.pth')
    val_losses_path = os.path.join(output_dir, 'val_losses.pth')

    train_losses = torch.load(train_losses_path)
    val_losses = torch.load(val_losses_path)

    # Plotting the loss curves
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.savefig(os.path.join(output_dir, f"train_val_loss.png"))
    plt.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default="paip", help='name of the dataset.')
    parser.add_argument('--data_dir', default="./dataset/paip/output_images_and_masks", 
                        help='base path of dataset.')
    parser.add_argument('--resolution', default=512, type=int,
                        help='resolution of img.')
    parser.add_argument('--pretrain', default="sam", type=str,
                        help='Use SAM pretrained weigths.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./apt",
                        help='save visualized and loss filename')
    args = parser.parse_args()
    
    args.world_size = int(os.environ['SLURM_NTASKS'])
    print(args.world_size)

    local_rank = int(os.environ['SLURM_LOCALID'])
    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME']) #str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    print("MASTER_ADDR:{}, MASTER_PORT:{}, WORLD_SIZE:{}, WORLD_RANK:{}, local_rank:{}".format(os.environ['MASTER_ADDR'], 
                                                    os.environ['MASTER_PORT'], 
                                                    os.environ['WORLD_SIZE'], 
                                                    os.environ['RANK'],
                                                    local_rank))
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=int(os.environ['RANK'])                                               
    )
    log(args=args)
    print("SLURM_LOCALID/lcoal_rank:{}, dist_rank:{}".format(local_rank, dist.get_rank()))

    print(f"Start running basic DDP example on rank {local_rank}.")
    device_id = local_rank % torch.cuda.device_count()


    main(args, device_id)
    
    dist.destroy_process_group()
