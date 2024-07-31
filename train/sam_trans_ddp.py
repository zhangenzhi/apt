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

from model.sam import SAMQDT
from apt.quadtree import FixedQuadTree
from dataset.paip_trans import PAIPTrans
from utils.focal_loss import DiceBLoss
from utils.draw import sub_miccai_plot

import logging

def dice_score(inputs, targets, smooth=1):    
    
    #flatten label and prediction tensors
    pred = torch.flatten(inputs[:,1:,:,:])
    true = torch.flatten(targets[:,1:,:,:])
    
    intersection = (pred * true).sum()
    coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)   
    return coeff  

def dice_score_plot(inputs, targets, smooth=1):     
    #flatten label and prediction tensors
    pred = inputs[...,0].flatten()
    true = targets[...,0].flatten()
    
    intersection = (pred * true).sum()
    coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)   
    return coeff  

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
    sqrt_len=int(math.sqrt(args.fixed_length))
    num_class = 2 
    
    model = SAMQDT(image_shape=(patch_size*sqrt_len, patch_size*sqrt_len),
            patch_size=args.patch_size,
            output_dim=num_class, 
            pretrain=args.pretrain,
            qdt=True)
    criterion = DiceBLoss()
    best_val_score = 0.0
    
    # Move the model to GPU
    model.to(device_id)
    if args.reload:
        if os.path.exists(os.path.join(args.savefile, "best_score_model.pth")):
            model.load_state_dict(torch.load(os.path.join(args.savefile, "best_score_model.pth")))
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Define the learning rate scheduler
    milestones =[int(args.epoch*r) for r in [0.5, 0.75, 0.875]]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Split the dataset into train, validation, and test sets
    data_path = args.data_dir
    dataset = PAIPTrans(data_path, args.resolution, fixed_length=args.fixed_length, patch_size=patch_size, normalize=False)
    # eval_set = MICCAIDataset(data_path, args.resolution, normalize=True, eval_mode=True)
    dataset_size = len(dataset)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size
    test_size = val_size
    logging.info("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, test_size))
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, dataset_size))
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,  sampler=val_sampler)
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
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                image, qimages, mask, qmasks, qdt_info, qdt_value = batch
                qimages = torch.reshape(qimages, shape=(-1,3,patch_size*sqrt_len, patch_size*sqrt_len))
                qmasks = torch.reshape(qmasks, shape=(-1,num_class,patch_size*sqrt_len, patch_size*sqrt_len))
                qimages, qmasks = qimages.to(device_id), qmasks.to(device_id)  # Move data to GPU
                optimizer.zero_grad()
                outputs = model(qimages)
                outputs = F.sigmoid(outputs)
                loss = criterion(outputs, qmasks, act=False)
                loss.backward()
                optimizer.step()
                # print("train step loss:{}, sec/step:{}".format(loss, (time.time()-start_time)/step))
                epoch_train_loss += loss.item()
                step+=1
        end_time = time.time()
        logging.info("epoch cost:{}, sec/img:{}, lr:{}".format(end_time-start_time, (end_time-start_time)/train_size, optimizer.param_groups[0]['lr']))

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
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        image, qimages, mask, qmasks, qdt_info, qdt_value = batch
                        seq_shape = qmasks.shape
                        qimages = torch.reshape(qimages, shape=(-1,3,patch_size*sqrt_len, patch_size*sqrt_len))
                        qmasks = torch.reshape(qmasks, shape=(-1,num_class,patch_size*sqrt_len, patch_size*sqrt_len))
                        qimages, qmasks = qimages.to(device_id), qmasks.to(device_id)  # Move data to GPU
                        outputs = model(qimages)
                        outputs = F.sigmoid(outputs)
                        loss = criterion(outputs, qmasks, act=False)
                        score = dice_score(outputs, qmasks)
                        epoch_val_loss += loss.item()
                        epoch_val_score += score.item()
            epoch_val_loss /= len(val_loader)
            epoch_val_score /= len(val_loader)

            # Visualize
            if (epoch - 1) % 10 == 9:  # Adjust the frequency of visualization
                with torch.no_grad():
                    for bi,batch in enumerate(val_loader):
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = torch.reshape(outputs, seq_shape)
                            qmasks = torch.reshape(qmasks, seq_shape)
                            qdt_score, qmask_score = sub_trans_plot(image, mask, qmasks=qmasks, pred_mask=outputs, qdt_info=qdt_info, 
                                                        fixed_length=args.fixed_length, bi=bi, epoch=epoch, output_dir=args.savefile)
                            epoch_qdt_score += qdt_score.item()
                            epoch_qmask_score += qmask_score.item()
            epoch_qdt_score /= len(val_loader)
            epoch_qmask_score /= len(val_loader)
            
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
    if device_id == 0:
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _, qimages, _, qmasks, _, qdt_value = batch
                    qimages, qmasks = qimages.to(device_id), qmasks.to(device_id)  # Move data to GPU
                    outputs = model(qimages)
                    outputs = F.sigmoid(outputs)
                    loss = criterion(outputs, qmasks, act=False)
                    test_loss += loss.item()

        test_loss /= len(test_loader)
        logging.info(f"Test Loss: {test_loss:.4f}")
        # draw_loss(output_dir=output_dir)

def sub_trans_plot(image, mask, qmasks, pred_mask, qdt_info, fixed_length, bi, epoch, output_dir):
    true_score = 0 
    best_score = 0
    for i in range(image.size(0)):
        image = image[i].cpu().permute(1, 2, 0).numpy()
        mask_true = mask[i].cpu().numpy()

        qmasks = (qmasks[i].cpu() > 0.5).numpy()
        qmasks.astype(np.int32)
        qmasks = qmasks[1]
        patch_size = qmasks.shape[0]
        qmasks = np.reshape(qmasks, (fixed_length, patch_size, patch_size))
        qmasks = np.repeat(np.expand_dims(qmasks, axis=-1), 3, axis=-1)
 
        # Squeeze the singleton dimension from mask_true
        mask_true = mask_true[1]
        mask_true = np.repeat(np.expand_dims(mask_true, axis=-1), 3, axis=-1)
        
        # print(mask_true.sum())
        pred_mask = (pred_mask[i].cpu() > 0.5).numpy()
        mask_pred = pred_mask[1]
        patch_size = mask_pred.shape[0]
        mask_pred = np.reshape(mask_pred, (fixed_length, patch_size, patch_size))
        mask_pred = np.repeat(np.expand_dims(mask_pred, axis=-1), 3, axis=-1)
      
        meta_info = []
        for nodes in qdt_info:
            n = []
            for idx in range(len(nodes)):
                n.append(nodes[idx][i].numpy())
            meta_info.append(n)
        
        qdt = FixedQuadTree(domain=mask_true, fixed_length=fixed_length, build_from_info=True, meta_info=meta_info)
        deoced_mask_pred = qdt.deserialize(seq=mask_pred, patch_size=patch_size, channel=3)
        decode_qmask = qdt.deserialize(seq=qmasks, patch_size=patch_size, channel=3)
        
        true_score += dice_score_plot(mask_true, targets=deoced_mask_pred)
        best_score += dice_score_plot(mask_true, targets=decode_qmask)
        
        mask_true = mask_true.astype(np.float64)

        # Plot and save images
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Input Image")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_true, cmap='gray')
        plt.title("True Mask")

        plt.subplot(1, 3, 3)
        plt.imshow(deoced_mask_pred, cmap='gray')
        plt.title("Predicted Mask")
        plt.savefig(os.path.join(output_dir, f"epoch_{epoch + 1}_sample_{bi + 1}.png"))
        plt.close()
        # true_score /= image.size(0)
        return true_score, best_score
     
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
