import os
import sys
sys.path.append("./")
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

from apt.quadtree import FixedQuadTree
from model.sam import SAMQDT
from dataset.s8d_2d import S8DFinetune2DAP
from utils.draw import draw_loss, sub_paip_plot
from utils.focal_loss import MulticlassDiceLoss

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def dice_score(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    eps: float = 1e-7,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute Dice score for multi-class segmentation.
    
    Args:
        inputs: (N, C, H, W) tensor of logits/probabilities
        targets: (N, C, H, W) one-hot encoded targets OR (N, H, W) class indices
        smooth: Laplace smoothing factor
        eps: Numerical stability term
        reduction: "mean"|"none"|"sum"
    
    Returns:
        Dice score (scalar or per-class scores)
    """
    # Convert targets to one-hot if needed
    if targets.dim() == 3:
        targets = torch.eye(inputs.shape[1], device=targets.device)[targets].permute(0,3,1,2)
    
    # Normalize inputs if needed (assumes inputs are logits)
    if inputs.size(1) > 1:
        probs = torch.softmax(inputs, dim=1)
    else:
        probs = torch.sigmoid(inputs)
    
    # Compute intersection and union
    dims = (0, 2, 3)  # Batch and spatial dims
    intersection = torch.sum(probs * targets, dim=dims)
    cardinality = torch.sum(probs + targets, dim=dims)
    
    # Compute dice per class
    dice = (2. * intersection + smooth) / (cardinality + smooth + eps)
    
    if reduction == "mean":
        return dice.mean()
    elif reduction == "sum":
        return dice.sum()
    return dice  # per-class scores  

import logging

# Configure logging
def log(args):
    os.makedirs(args.savefile, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.savefile, "out.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
def main(args):
    
    log(args=args)
    # Create an instance of the U-Net model and other necessary components
    patch_size=args.patch_size
    sqrt_len=int(math.sqrt(args.fixed_length))
    num_class = 5
    
    model = SAMQDT(image_shape=(patch_size*sqrt_len, patch_size*sqrt_len),
            patch_size=args.patch_size,
            output_dim=num_class, 
            pretrain=args.pretrain,
            qdt=True)
    criterion = MulticlassDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    best_val_score = 0.0
    
    # Move the model to GPU
    model.to(device)
    if args.reload:
        if os.path.exists(os.path.join(args.savefile, "best_score_model.pth")):
            model.load_state_dict(torch.load(os.path.join(args.savefile, "best_score_model.pth")))
    # Define the learning rate scheduler
    milestones =[int(args.epoch*r) for r in [0.5, 0.75, 0.875]]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Split the dataset into train, validation, and test sets
    data_path = args.data_dir
    dataset = S8DFinetune2DAP(data_path, num_classes=num_class, fixed_length=args.fixed_length, patch_size=patch_size)
    dataset_size = len(dataset)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size
    test_size = val_size
    logging.info("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, test_size))
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, dataset_size))
    train_set = Subset(dataset, train_indices)
    val_set = test_set = Subset(dataset, val_indices)
    # train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Training loop
    num_epochs = args.epoch
    train_losses = []
    val_losses = []
    output_dir = args.savefile  # Change this to the desired directory
    os.makedirs(output_dir, exist_ok=True)
    import time
    import random
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        start_time = time.time()
        step=1
        for batch in train_loader:
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            qimages, qmasks, seq_size, seq_pos = batch
            qimages = torch.reshape(qimages,shape=(-1,3,patch_size*sqrt_len, patch_size*sqrt_len))
            qmasks = torch.reshape(qmasks,shape=(-1,num_class,patch_size*sqrt_len, patch_size*sqrt_len))
            qimages, qmasks = qimages.to(device), qmasks.to(device)  # Move data to GPU
        
            outputs = model(qimages)
            loss = criterion(outputs, qmasks)
            score = dice_score(outputs, qmasks)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_train_loss += loss.item()
            step+=1
        end_time = time.time()
        logging.info("epoch cost:{}, sec/img:{}, lr:{}".format(end_time-start_time, (end_time-start_time)/train_size, optimizer.param_groups[0]['lr']))
        logging.info("train step loss:{}, train step score:{}, sec/step:{}".format(loss, score, (time.time()-start_time)/step))

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        scheduler.step()

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_score = 0.0
        epoch_qdt_score = 0.0
        epoch_qmask_score = 0.0
        with torch.no_grad():
            for bi,batch in enumerate(val_loader):
                # with torch.autocast(device_type='cuda', dtype=torch.float16):
                qimages, qmasks, seq_size, seq_pos = batch
                seq_shape = qmasks.shape
                qimages = torch.reshape(qimages,shape=(-1,3,patch_size*sqrt_len, patch_size*sqrt_len))
                qmasks = torch.reshape(qmasks,shape=(-1,num_class,patch_size*sqrt_len, patch_size*sqrt_len))
                qimages, qmasks = qimages.to(device), qmasks.to(device)  # Move data to GPU
                outputs = model(qimages)
                loss = criterion(outputs, qmasks)
                score = dice_score(outputs, qmasks)
                # if  (epoch - 1) % 10 == 9:  # Adjust the frequency of visualization
                #     outputs = torch.reshape(outputs, seq_shape)
                #     qmasks = torch.reshape(qmasks, seq_shape)
                #     qdt_score, qmask_score = sub_trans_plot(image, mask, qmasks=qmasks, pred_mask=outputs, qdt_info=qdt_info, 
                #                                fixed_length=args.fixed_length, bi=bi, epoch=epoch, output_dir=args.savefile)
                #     epoch_qdt_score += qdt_score.item()
                #     epoch_qmask_score += qmask_score.item()
                epoch_val_loss += loss.item()
                epoch_val_score += score.item()

        epoch_val_loss /= len(val_loader)
        epoch_val_score /= len(val_loader)
        epoch_qdt_score /= len(val_loader)
        epoch_qmask_score /= len(val_loader)
        val_losses.append(epoch_val_loss)
        # Save the best model based on validation accuracy
        if epoch_val_score > best_val_score:
            best_val_score = epoch_val_score
            torch.save(model.state_dict(), os.path.join(args.savefile, "best_score_model.pth"))
            logging.info(f"Model save with dice score {best_val_score} at epoch {epoch}")
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f},\
            Score: {epoch_val_score:.4f} QDT Score: {epoch_qdt_score:.4f}/{epoch_qmask_score:.4f}.")

    # Save train and validation losses
    train_losses_path = os.path.join(output_dir, 'train_losses.pth')
    val_losses_path = os.path.join(output_dir, 'val_losses.pth')
    torch.save(train_losses, train_losses_path)
    torch.save(val_losses, val_losses_path)

    # Test the model
    model.eval()
    test_loss = 0.0
    epoch_test_score = 0
    with torch.no_grad():
        for batch in test_loader:
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, qimages, _, qmasks, _, qdt_value = batch
            qimages = torch.reshape(qimages, shape=(-1,3,patch_size*sqrt_len, patch_size*sqrt_len))
            qmasks = torch.reshape(qmasks, shape=(-1,num_class,patch_size*sqrt_len, patch_size*sqrt_len))
            qimages, qmasks = qimages.to(device), qmasks.to(device)  # Move data to GPU
            outputs = model(qimages)
            loss = criterion(outputs, qmasks)
            score = dice_score(outputs, qmasks)
            test_loss += loss.item()
            epoch_test_score += score.item()

    test_loss /= len(test_loader)
    epoch_test_score /= len(test_loader)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Score: {epoch_test_score:.4f}")
    draw_loss(output_dir=output_dir)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="s8d", 
                        help='base path of dataset.')
    parser.add_argument('--data_dir', default="/lustre/orion/nro108/world-shared/enzhi/Riken_XCT_Simulated_Data/8192x8192_2d_Simulations/Noise_0.05_Blur_2_sparsity_2_NumAng_3600", 
                        help='base path of dataset.')
    parser.add_argument('--fixed_length', default=10201, type=int,
                        help='length of sequence.')
    parser.add_argument('--patch_size', default=8, type=int,
                        help='patch size.')
    parser.add_argument('--pretrain', default="sam-b", type=str,
                        help='Use SAM pretrained weigths.')
    parser.add_argument('--reload', default=True, type=bool,
                        help='Use SAM pretrained weigths.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./sap_s8d",
                        help='save visualized and loss filename')
    args = parser.parse_args()

    main(args)
