import os
import sys
sys.path.append("./")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def sub_paip_plot(model, eval_loader, epoch, device, output_dir):
    # Visualize and save predictions on a few validation samples
        model.eval()
        for bi,batch in enumerate(eval_loader):
            with torch.no_grad():
                _, qsample_images, _, qsample_masks= batch
                qsample_images = torch.reshape(qsample_images,shape=(-1, 3, 8*16, 8*16))
                qsample_masks = torch.reshape(qsample_masks,shape=(-1, 2, 8*16, 8*16))
                qsample_images, qsample_masks = qsample_images.to(device), qsample_masks.to(device)  # Move data to GPU
                outputs = model(qsample_images)
                qsample_outputs = torch.sigmoid(outputs)

                for i in range(qsample_images.size(0)):
                    image = qsample_images[i].cpu().permute(1, 2, 0).numpy()
                    mask_true = qsample_masks[i].cpu().numpy()
                    mask_pred = (qsample_outputs[i].cpu() > 0.5).numpy()
                    
                    # Squeeze the singleton dimension from mask_true
                    mask_true = mask_true[1]
                    mask_pred=mask_pred[1]

                    # Plot and save images
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(image)
                    plt.title("Input Image")

                    plt.subplot(1, 3, 2)
                    plt.imshow(mask_true, cmap='gray')
                    plt.title("True Mask")

                    plt.subplot(1, 3, 3)
                    plt.imshow(mask_pred, cmap='gray')
                    plt.title("Predicted Mask")
                    plt.savefig(os.path.join(output_dir, f"epoch_{epoch + 1}_sample_{bi + 1}.png"))
                    plt.close()
                    
def sub_miccai_plot(model, eval_loader, epoch, device, output_dir):
    # Visualize and save predictions on a few validation samples
        model.eval()
        for bi,batch in enumerate(eval_loader):
            with torch.no_grad():
                qsample_images, qsample_masks= batch
                qsample_images = torch.reshape(qsample_images,shape=(-1, 3, 2048, 2048))
                qsample_masks = torch.reshape(qsample_masks,shape=(-1, 2,  2048, 2048))
                qsample_images, qsample_masks = qsample_images.to(device), qsample_masks.to(device)  # Move data to GPU
                outputs = model(qsample_images)
                qsample_outputs = torch.sigmoid(outputs)

                for i in range(qsample_images.size(0)):
                    image = qsample_images[i].cpu().permute(1, 2, 0).numpy()
                    mask_true = qsample_masks[i].cpu().numpy()
                    mask_pred = (qsample_outputs[i, 0].cpu() > 0.5).numpy()
                    
                    # Squeeze the singleton dimension from mask_true
                    mask_true = np.squeeze(mask_true, axis=0)

                    # Plot and save images
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(image)
                    plt.title("Input Image")

                    plt.subplot(1, 3, 2)
                    plt.imshow(mask_true, cmap='gray')
                    plt.title("True Mask")

                    plt.subplot(1, 3, 3)
                    plt.imshow(mask_pred, cmap='gray')
                    plt.title("Predicted Mask")
                    
                    basedir = os.path.join(output_dir, bi)
                    os.makedirs(basedir, exist_ok=True)
                    plt.savefig(os.path.join(basedir, f"epoch_{epoch + 1}_sample_{i + 1}.png"))
                    plt.close()
                    return
          
          
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