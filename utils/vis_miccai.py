import os
import sys
from PIL import Image
import numpy as np
sys.path.append("./")
from collections import OrderedDict

import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from model.sam import SAM
from torch.utils.data import DataLoader
from dataset.miccai import MICCAIDataset
from train.sam_miccai import DiceBCELoss

def save_predicts(outputs, resolution, filename):
    outputs = outputs.to('cpu')
    save_name=f"predict_patches-{resolution}"
    for i,fp in enumerate(filename):
        pardir = os.path.dirname(os.path.dirname(fp))
        save_path = os.path.join(pardir, save_name)
        os.makedirs(save_path, exist_ok=True)
        basename = os.path.basename(fp)
        save_path = os.path.join(save_path,basename)
        save_image(outputs[i], save_path)

def main(path, model_weights, resolution, batch_size, patch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = DiceBCELoss().to(device)
    val_score = 0.0
    val_loss = 0.0
    
    model = SAM(image_shape=(resolution, resolution),
        patch_size=patch_size,
        output_dim=1, 
        pretrain="sam-b")
    
    # def fix_model_state_dict(state_dict):
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k
    #         if name.startswith('module.'):
    #             name = name[7:]  # remove 'module.' of dataparallel
    #         new_state_dict[name] = v
    #     return new_state_dict
    # model.load_state_dict(fix_model_state_dict(torch.load(os.path.join(model_weights, "best_score_model.pth"))))
    model.load_state_dict(torch.load(os.path.join(model_weights, "best_score_model.pth")))
    
    model.to(device)
    
    dataset = MICCAIDataset(path, resolution, normalize=False, eval_mode=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=False)
    dataset_size= len(dataset)
    opt = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    
    # model.eval()
    # with torch.no_grad():
    for i,batch in enumerate(data_loader):
        model.train()
        images, masks = batch
        images, masks = images.to(device), masks.to(device)  # Move data to GPU
        outputs = model(images)
        opt.zero_grad()
        loss, score = criterion(outputs, masks)
        loss.backward()
        opt.step()
        pred_outputs = torch.sigmoid(outputs)
        print(f"Shape:{pred_outputs.shape} Mean pred:{torch.mean(pred_outputs)} Mean true:{torch.mean(masks)}")

        print(f"score:{score}, step:{i*batch_size}")
        val_score += score
        val_loss += loss
        
        # filename = data_loader.dataset.image_filenames[i*batch_size:min((i+1)*batch_size, dataset_size)]
        # save_name = f"predict_patches-{resolution}"
        
        # for i, fp in enumerate(filename):
        #     # import pdb
        #     # pdb.set_trace()
        #     mask_pred = (pred_outputs[i, 0].cpu() > 0.5).numpy()
        #     pardir = os.path.dirname(os.path.dirname(fp))
        #     save_path = os.path.join(pardir, save_name)
        #     os.makedirs(save_path, exist_ok=True)
        #     basename = os.path.basename(fp)
        #     save_path = os.path.join(save_path, basename)
        #     plt.imsave(save_path, mask_pred, cmap='gray')
        
    print("Total mean score:{}, loss:{}".format(val_score/len(data_loader), val_loss/len(data_loader)))

if __name__ == "__main__":
    main(path="/lustre/orion/bif146/world-shared/enzhi/miccai_patches", 
        model_weights="/lustre/orion/bif146/world-shared/enzhi/apt/sam-b_miccai-n32-pz8-bz4-vis/",
        patch_size=8,
        batch_size=4,
        resolution=512)