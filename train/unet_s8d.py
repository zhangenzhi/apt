import os

import sys
sys.path.append("./")
import argparse

import torch
import torch.optim as optim

from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from model.unet import Unet, LightweightUNet
from dataset.s8d_2d import S8DFinetune2D
from utils.focal_loss import MulticlassDiceLoss
from utils.draw import draw_loss

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
    num_class = 5
    
    # model = Unet(n_class=num_class, in_channels=1, pretrained=True)
    model = LightweightUNet(n_class=num_class, in_channels=1)
    criterion = MulticlassDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    best_val_score = 0.0
    
    # Move the model to GPU
    model = model.to(device)
    model = nn.DataParallel(model)
    if args.reload:
        if os.path.exists(os.path.join(args.savefile, "best_score_model.pth")):
            model.load_state_dict(torch.load(os.path.join(args.savefile, "best_score_model.pth")))
    # Define the learning rate scheduler
    milestones =[int(args.epoch*r) for r in [0.5, 0.75, 0.875]]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Split the dataset into train, validation, and test sets
    data_path = args.data_dir
    dataset = S8DFinetune2D(data_path, num_classes=num_class)
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

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=32, shuffle=True)
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
        for batch in train_loader:
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            images, masks, _ = batch
            images, masks = images.to(device), masks.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            print(f"Train Step Loss: {loss}, time cost: {time.time() - start_time}")
                
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
            for bi,batch in enumerate(val_loader):
                images, masks, _ = batch
                images, masks = images.to(device), masks.to(device)  # Move data to GPU
                outputs = model(images)
                loss = criterion(outputs, masks)
                score = dice_score(outputs, masks)
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
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Score: {epoch_val_score:.4f}")
        
        if  (epoch - 1) % 10 == 9:  # Adjust the frequency of visualization
            sub_trans_plot(images, masks, pred=outputs, bi=bi, epoch=epoch, output_dir=args.savefile)

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
            images, masks, _ = batch
            images, masks = images.to(device), masks.to(device)  # Move data to GPU
            outputs = model(images)
            loss = criterion(outputs, masks)
            score = dice_score(outputs, masks)
            test_loss += loss.item()
            epoch_test_score += score.item()

    test_loss /= len(test_loader)
    epoch_test_score /= len(test_loader)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Score: {epoch_test_score:.4f}")
    draw_loss(output_dir=output_dir)

def sub_trans_plot(image, mask, pred, bi, epoch, output_dir):
    # only one sample
    
    # import pdb;pdb.set_trace()
    
    image = image[0]
    true_mask = mask[0]
    pred_mask = pred[0]
  
    filename_image = f"image_epoch_{epoch + 1}_sample_{bi + 1}.tiff"
    filename_mask = f"mask_epoch_{epoch + 1}_sample_{bi + 1}.png"
    filename_pred = f"pred_epoch_{epoch + 1}_sample_{bi + 1}.png"

    from dataset.utilz import save_input_as_image, save_pred_as_mask
    
    save_input_as_image(image, os.path.join(output_dir,filename_image))
    save_pred_as_mask(true_mask, os.path.join(output_dir,filename_mask))
    save_pred_as_mask(pred_mask, os.path.join(output_dir,filename_pred))
    print(f"Visualized for {epoch}-{bi}, Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default="paip", help='name of the dataset.')
    parser.add_argument('--data_dir', default="/lustre/orion/nro108/world-shared/enzhi/Riken_XCT_Simulated_Data/8192x8192_2d_Simulations/Noise_0.05_Blur_2_sparsity_2_NumAng_3600", 
                        help='base path of dataset.')
    parser.add_argument('--reload', default=True, type=bool,
                        help='Use SAM pretrained weigths.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./unet_output",
                        help='save visualized and loss filename')
    args = parser.parse_args()

    main(args)
