import os
import sys
sys.path.append("./")
import argparse

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

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from apt.quadtree import FixedQuadTree
from model.sam import SAMQDT
from dataset.s8d_2d import S8DFinetune2DAP,collate_fn
from utils.draw import draw_loss, sub_paip_plot
from utils.focal_loss import DiceCELoss

import logging

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

# Configure logging
def log(args):
    os.makedirs(args.savefile, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.savefile, "out.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
def main(args, device_id):
    log(args=args)
    
    # Create an instance of the U-Net model and other necessary components
    patch_size=args.patch_size
    num_class = 5
    
    model = SAMQDT(image_shape=(1, args.fixed_length),
            patch_size=args.patch_size,
            output_dim=num_class,
            in_chans = 1, 
            pretrain=args.pretrain,
            # pretrain=False,
            qdt=True, use_qdt_pos=True, linear_embed=True)
    criterion = DiceCELoss()
    best_val_score = 0.0
    
    # Move the model to GPU
    model.to(device_id)
    if args.reload:
        if os.path.exists(os.path.join(args.savefile, "best_score_model.pth")):
            model.load_state_dict(torch.load(os.path.join(args.savefile, "best_score_model.pth")))
    model = DDP(model, device_ids=[device_id], find_unused_parameters=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
    # logging.info("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, test_size))
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, dataset_size))
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,  sampler=val_sampler, collate_fn=collate_fn)
    test_loader = val_loader

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
        train_loader.sampler.set_epoch(epoch)
        start_time = time.time()
        step=1
        for batch in train_loader:
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            image, mask, qimages, qmasks, qdt, seq_size, seq_pos = batch # torch.Size([1, 5, 10201, 64])
            qimages, qmasks = qimages.to(device_id), qmasks.to(device_id)  # Move data to GPU
            seq_size, seq_pos = seq_size.to(device_id), seq_pos.to(device_id)
            seq_size= seq_size.unsqueeze(-1)
            seq_ps = torch.concat([seq_size, seq_pos],dim=-1)
            
            optimizer.zero_grad()
            outputs = model(qimages, seq_ps)
            loss = criterion(outputs, qmasks)
            score = dice_score(outputs, qmasks)
            
            loss.backward()
            optimizer.step()
            # print("train step loss:{}, sec/step:{}".format(loss, (time.time()-start_time)/step))
            epoch_train_loss += loss.item()
            step+=1
        end_time = time.time()
        logging.info("epoch cost:{}, sec/img:{}, lr:{}".format(end_time-start_time, (end_time-start_time)/train_size, optimizer.param_groups[0]['lr']))
        logging.info("train step loss:{}, train step score:{}, sec/step:{}".format(loss, score, (time.time()-start_time)/step))
        with torch.no_grad():
            if (epoch - 1) % 10 == 9 and device_id == 0:  # Adjust the frequency of visualization
                sub_trans_plot(image, mask, qmasks=qmasks, pred=outputs, qdt=qdt, 
                                fixed_length=args.fixed_length, bi=-1, epoch=epoch, output_dir=args.savefile)

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        scheduler.step()

        if device_id == 0:
            # Validation
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_score = 0.0
            epoch_qdt_score = 0.0
            epoch_qmask_score = 0.0
            with torch.no_grad():
                for bi,batch in enumerate(val_loader):
                    # with torch.autocast(device_type='cuda', dtype=torch.float16):
                    image, mask, qimages, qmasks, qdt, seq_size, seq_pos = batch # torch.Size([1, 5, 10201, 64])
                    qimages, qmasks = qimages.to(device_id), qmasks.to(device_id)  # Move data to GPU
                    seq_size, seq_pos = seq_size.to(device_id), seq_pos.to(device_id)
                    seq_size= seq_size.unsqueeze(-1)
                    seq_ps = torch.concat([seq_size, seq_pos],dim=-1)
            
                    outputs = model(qimages, seq_ps)
                    loss = criterion(outputs, qmasks)
                    score = dice_score(outputs, qmasks)
                    epoch_val_loss += loss.item()
                    epoch_val_score += score.item()
            epoch_val_loss /= len(val_loader)
            epoch_val_score /= len(val_loader)

            # # Visualize
            if (epoch - 1) % 10 == 9:  # Adjust the frequency of visualization
                sub_trans_plot(image, mask, qmasks=qmasks, pred=outputs, qdt=qdt, 
                                fixed_length=args.fixed_length, bi=bi, epoch=epoch, output_dir=args.savefile)

            epoch_qdt_score /= len(val_loader)
            epoch_qmask_score /= len(val_loader)
            
            # Save the best model based on validation accuracy
            if epoch_val_score > best_val_score and dist.get_rank()==0:
                best_val_score = epoch_val_score
                torch.save(model.module.state_dict(), os.path.join(args.savefile, "best_score_model.pth"))
                logging.info(f"Model save with dice score {best_val_score} at epoch {epoch}")
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f},\
                Score: {epoch_val_score:.4f} QDT Score: {epoch_qdt_score:.4f}/{epoch_qmask_score:.4f}.")

                        
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
                # with torch.autocast(device_type='cuda', dtype=torch.float16):
                image, mask, qimages, qmasks, qdt = batch
                qimages, qmasks = qimages.to(device_id), qmasks.to(device_id)  # Move data to GPU
            
                outputs = model(qimages)
                loss = criterion(outputs, qmasks)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        logging.info(f"Test Loss: {test_loss:.4f}")
        # draw_loss(output_dir=output_dir)

def sub_trans_plot(image, mask, qmasks, pred, qdt, fixed_length, bi, epoch, output_dir):
    # only one sample
    
    image = image[0]
    image = image.squeeze().cpu().numpy()
    
    true_mask = mask[0]
    true_mask = true_mask.squeeze().cpu().numpy()
    
    true_seq_mask = qmasks[0]
    true_seq_mask = true_seq_mask.squeeze().cpu().numpy()
    
    pred_seq_mask = pred[0]
    pred_seq_mask = pred_seq_mask.squeeze().cpu().numpy()
    
    qdt = qdt[0]
    # import pdb;pdb.set_trace()
    decoded_true_mask = qdt.deserialize(seq=true_seq_mask, patch_size=8, channel=5)
    decoded_true_mask = np.transpose(decoded_true_mask, (2, 1, 0))
    decoded_pred_mask = qdt.deserialize(seq=pred_seq_mask, patch_size=8, channel=5)
    decoded_pred_mask = np.transpose(decoded_pred_mask, (2, 1, 0)) 
    # import pdb;pdb.set_trace()
    
    filename_image = f"image_epoch_{epoch + 1}_sample_{bi + 1}.tiff"
    filename_mask = f"mask_epoch_{epoch + 1}_sample_{bi + 1}.png"
    # filename_pred = f"pred_epoch_{epoch + 1}_sample_{bi + 1}.png"
    filename_decoded_mask = f"decoded_mask_epoch_{epoch + 1}_sample_{bi + 1}.png"
    filename_decoded_pred = f"decoded_pred_epoch_{epoch + 1}_sample_{bi + 1}.png"

    from dataset.utilz import save_input_as_image, save_pred_as_mask
    
    save_input_as_image(image, os.path.join(output_dir, filename_image))
    save_pred_as_mask(true_mask, os.path.join(output_dir, filename_mask))
    save_pred_as_mask(decoded_true_mask, os.path.join(output_dir, filename_decoded_mask))
    save_pred_as_mask(decoded_pred_mask, os.path.join(output_dir, filename_decoded_pred))
    print(f"Visualized for {epoch}-{bi}, Done!")
    
     
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
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='resolution of img.')
    parser.add_argument('--patch_size', default=8, type=int,
                        help='patch size.')
    parser.add_argument('--fixed_length', default=512, type=int,
                        help='length of sequence.')
    parser.add_argument('--pretrain', default="sam-b", type=str,
                        help='Use SAM pretrained weigths.')
    parser.add_argument('--reload', default=True, type=bool,
                        help='Use reload val weigths.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./apt",
                        help='save visualized and loss filename')
    args = parser.parse_args()

    args.world_size = int(os.environ['SLURM_NTASKS'])
    
    log(args=args)
    
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
    print("SLURM_LOCALID/lcoal_rank:{}, dist_rank:{}".format(local_rank, dist.get_rank()))

    print(f"Start running basic DDP example on rank {local_rank}.")
    device_id = local_rank % torch.cuda.device_count()
    main(args, device_id)
    
    dist.destroy_process_group()
