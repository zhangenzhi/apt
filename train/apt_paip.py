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

from model.apt import APT
# from model.unet import Unet
from dataset.paip_qdt import PAIPQDTDataset
# from dataset.paip_mqdt import PAIPQDTDataset

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
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = self.weight*BCE + (1-self.weight)*dice_loss
        
        return Dice_BCE
    
def main(args):
    # Create an instance of the U-Net model and other necessary components
    # model = Unet(n_class=1)
    model = APT(qdt_shape=(args.patch_size, args.fixed_length*args.patch_size),
            input_dim=3, 
            output_dim=1, 
            embed_dim=768,
            num_heads=12,
            pretrain=args.pretrain, 
            dropout=0.1)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    # Move the model to GPU
    model.to(device)
    # Define the learning rate scheduler
    # milestones =[int(args.epoch*r) for r in [0.5, 0.75, 0.875]]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Split the dataset into train, validation, and test sets
    data_path = Path(args.data_dir)
    dataset = PAIPQDTDataset(data_path, args.resolution, args.fixed_length, args.patch_size, normalize=True)
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = (dataset_size - train_size) // 2
    test_size = dataset_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

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
        for batch in train_loader:
            images, qdt_img, masks, qdt_mask = batch
            qdt_img, qdt_mask = qdt_img.to(device), qdt_mask.to(device)  # Move data to GPU
            optimizer.zero_grad()

            outputs = model(qdt_img)
            loss = criterion(outputs, qdt_mask)
            # print("train step loss:{}".format(loss))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
        end_time = time.time()
        print("epoch cost:{}, sec/img:{}".format(end_time-start_time,(end_time-start_time)/train_size))

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        # scheduler.step()

        # Validation
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images, qdt_img, masks, qdt_mask = batch
                qdt_img, qdt_mask = qdt_img.to(device), qdt_mask.to(device)  # Move data to GPU
                outputs = model(qdt_img)
                loss = criterion(outputs, qdt_mask)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

        # Visualize and save predictions on a few validation samples
        if (epoch + 1) % 3 == 0:  # Adjust the frequency of visualization
            model.eval()
            with torch.no_grad():
                sample_images, sample_qdt_img, sample_masks,  sample_qdt_masks= next(iter(val_loader))
                sample_qdt_img, sample_qdt_masks = sample_qdt_img.to(device), sample_qdt_masks.to(device)  # Move data to GPU
                sample_outputs = torch.sigmoid(model(sample_qdt_img))

                for i in range(sample_images.size(0)):
                    image = sample_images[i].cpu().permute(1, 2, 0).numpy()
                    mask_true = sample_qdt_masks[i].cpu().numpy()
                    mask_pred = (sample_outputs[i, 0].cpu() > 0.5).numpy()
                    
                    # Squeeze the singleton dimension from mask_true
                    mask_true = np.squeeze(mask_true, axis=0)

                    # # Plot and save images
                    # plt.figure(figsize=(12, 4))
                    # plt.subplot(1, 3, 1)
                    # plt.imshow(image)
                    # plt.title("Input Image")

                    # plt.subplot(1, 3, 2)
                    # plt.imshow(mask_true, cmap='gray')
                    # plt.title("True Mask")

                    # plt.subplot(1, 3, 3)
                    # plt.imshow(mask_pred, cmap='gray')
                    # plt.title("Predicted Mask")

                    # plt.savefig(os.path.join(output_dir, f"epoch_{epoch + 1}_sample_{i + 1}.png"))
                    # plt.close()

    # Save train and validation losses
    train_losses_path = os.path.join(output_dir, 'train_losses.pth')
    val_losses_path = os.path.join(output_dir, 'val_losses.pth')
    torch.save(train_losses, train_losses_path)
    torch.save(val_losses, val_losses_path)

    # Test the model
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            images, qdt_img, masks, qdt_mask = batch
            qdt_img, qdt_mask = qdt_img.to(device), qdt_mask.to(device)  # Move data to GPU
            outputs = model(qdt_img)
            loss = criterion(outputs, qdt_mask)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
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
    parser.add_argument('--patch_size', default=8, type=int,
                        help='patch size.')
    parser.add_argument('--fixed_length', default=1024, type=int,
                        help='length of sequence.')
    parser.add_argument('--pretrain', default="sam", type=str,
                        help='Use SAM pretrained weigths.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./apt",
                        help='save visualized and loss filename')
    args = parser.parse_args()

    main(args)
