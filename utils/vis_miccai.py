import os
import sys
sys.path.append("./")
from collections import OrderedDict

import torch
from torchvision.utils import save_image

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
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    criterion = DiceBCELoss().to(device)
    val_score = 0.0
    
    model = SAM(image_shape=(resolution, resolution),
        patch_size=patch_size,
        output_dim=1, 
        pretrain=False)
    
    def fix_model_state_dict(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        return new_state_dict
    model.load_state_dict(fix_model_state_dict(torch.load(os.path.join(model_weights, "best_score_model.pth"))))
    
    model.to(device)
    
    dataset = MICCAIDataset(path, resolution, normalize=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=False)
    dataset_size= len(dataset)

    model.eval()
    for i,batch in enumerate(data_loader):
        # import pdb
        # pdb.set_trace()
        
        images, masks = batch
        images, masks = images.to(device), masks.to(device)  # Move data to GPU
        outputs = model(images)
        loss, score = criterion(outputs, masks)
        print(f"score:{score}, step:{i*batch_size}")
        val_score += score
        
        # filename = data_loader.dataset.image_filenames[i*batch_size:min((i+1)*batch_size, dataset_size)]
        # save_name=f"predict_patches-{resolution}"
        
        # for i,fp in enumerate(filename):
        #     pardir = os.path.dirname(os.path.dirname(fp))
        #     save_path = os.path.join(pardir, save_name)
        #     os.makedirs(save_path, exist_ok=True)
        #     basename = os.path.basename(fp)
        #     save_path = os.path.join(save_path,basename)
            # save_image(outputs[i], save_path)
        
    print("Total mean score:{}".format(val_score/len(data_loader)))

if __name__ == "__main__":
    main(path="/lustre/orion/bif146/world-shared/enzhi/miccai_patches", 
        model_weights="/lustre/orion/bif146/world-shared/enzhi/apt/sam-b_miccai-n128-pz8-bz4/",
        patch_size=8,
        batch_size=1,
        resolution=512)