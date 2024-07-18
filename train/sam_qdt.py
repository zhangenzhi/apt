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

from model.apt import APT
from model.sam import SAMQDT
from model.unet import Unet
from dataset.paip_qdt import PAIPQDTDataset
from utils.draw import draw_loss, sub_paip_plot
from utils.focal_loss import DiceBCELoss, DiceBLoss
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
    
def main(args):
    
    log(args=args)
    # Create an instance of the U-Net model and other necessary components
    patch_size=args.patch_size
    sqrt_len=int(math.sqrt(args.fixed_length))
    
    num_class = 2 
    model = SAMQDT(image_shape=(patch_size*sqrt_len, patch_size*sqrt_len),
            patch_size=args.patch_size,
            output_dim=num_class, 
            pretrain=args.pretrain)
    criterion = DiceBLoss()
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
    dataset = PAIPQDTDataset(data_path, args.resolution, args.fixed_length, args.patch_size, sths=[1,3,5], normalize=False)
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

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
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
        # sub_paip_plot(model=model, eval_loader=val_loader, epoch=epoch, device=device, output_dir=args.savefile)
        # break
        model.train()
        epoch_train_loss = 0.0
        start_time = time.time()
        for batch in train_loader:
            _, qimages, _, qmasks = batch
            qimages = torch.reshape(qimages,shape=(-1,3,patch_size*sqrt_len, patch_size*sqrt_len))
            qmasks = torch.reshape(qmasks,shape=(-1,num_class,patch_size*sqrt_len, patch_size*sqrt_len))
            qimages, qmasks = qimages.to(device), qmasks.to(device)  # Move data to GPU
            
            optimizer.zero_grad()
            outputs = model(qimages)
            loss,_ = criterion(outputs, qmasks)
            # print("train step loss:{}".format(loss))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
        end_time = time.time()
        logging.info("epoch cost:{}, sec/img:{}".format(end_time-start_time,(end_time-start_time)/train_size))

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        scheduler.step()

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_score = 0.0
        with torch.no_grad():
            for batch in val_loader:
                _, qimages, _, qmasks = batch
                qimages = torch.reshape(qimages,shape=(-1,3,patch_size*sqrt_len, patch_size*sqrt_len))
                qmasks = torch.reshape(qmasks,shape=(-1,num_class,patch_size*sqrt_len, patch_size*sqrt_len))
                qimages, qmasks = qimages.to(device), qmasks.to(device)  # Move data to GPU
                outputs = model(qimages)
                loss, score = criterion(outputs, qmasks)
                epoch_val_loss += loss.item()
                epoch_val_score += score.item()

        epoch_val_loss /= len(val_loader)
        epoch_val_score /= len(val_loader)
        val_losses.append(epoch_val_loss)
        # Save the best model based on validation accuracy
        if epoch_val_score > best_val_score:
            best_val_score = epoch_val_score
            torch.save(model.state_dict(), os.path.join(args.savefile, "best_score_model.pth"))
            logging.info(f"Model save with dice score {best_val_score} at epoch {epoch}")
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Score: {epoch_val_score:.4f}.")

        # Visualize and save predictions on a few validation samples
        if (epoch + 1) % 3 == 1:  # Adjust the frequency of visualization
            sub_paip_plot(model=model, eval_loader=val_loader, epoch=epoch, device=device, output_dir=args.savefile)

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
            _, qimages, _, qmasks = batch
            qimages = torch.reshape(qimages, shape=(-1,3,patch_size*sqrt_len, patch_size*sqrt_len))
            qmasks = torch.reshape(qmasks, shape=(-1,num_class,patch_size*sqrt_len, patch_size*sqrt_len))
            qimages, qmasks = qimages.to(device), qmasks.to(device)  # Move data to GPU
            outputs = model(qimages)
            loss,score = criterion(outputs, qmasks)
            test_loss += loss.item()
            epoch_test_score += score.item()

    test_loss /= len(test_loader)
    epoch_test_score /= len(test_loader)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Score: {epoch_test_score:.4f}")
    draw_loss(output_dir=output_dir)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default="paip", help='name of the dataset.')
    parser.add_argument('--data_dir', default="./dataset/paip/output_images_and_masks", 
                        help='base path of dataset.')
    parser.add_argument('--resolution', default=1024, type=int,
                        help='resolution of img.')
    parser.add_argument('--fixed_length', default=512, type=int,
                        help='length of sequence.')
    parser.add_argument('--patch_size', default=8, type=int,
                        help='patch size.')
    parser.add_argument('--pretrain', default="sam", type=str,
                        help='Use SAM pretrained weigths.')
    parser.add_argument('--reload', default=True, type=bool,
                        help='Use SAM pretrained weigths.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./output",
                        help='save visualized and loss filename')
    args = parser.parse_args()

    main(args)
