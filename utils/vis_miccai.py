import os
import sys
sys.path.append("./")
import torch
from model.sam import SAM
from torch.utils.data import DataLoader
from dataset.miccai import MICCAIDataset
from train.sam_miccai import DiceBCELoss

def evaluate(model, dataset):
    pass

def merge(to_resolution, save_path):
    pass

def main(path, model_weights, resolution, patch_size, to_resolution, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = DiceBCELoss().to(device)
    val_score = 0.0
    
    model = SAM(image_shape=(resolution, resolution),
        patch_size=patch_size,
        output_dim=1, 
        pretrain="None")
    def fix_model_state_dict(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        return new_state_dict
    state_dict = torch.load(os.path.join(model_weights, "best_score_model.pth"))
    model.load_state_dict(fix_model_state_dict(state_dict))
    
    dataset = MICCAIDataset(path, resolution, normalize=False)
    data_loader = DataLoader(dataset, batch_size=8, num_workers=16, shuffle=False)
    
    import pdb
    pdb.set_trace()
    
    for i,batch in enumerate(data_loader):
        filename = data_loader.dataset.samples[i]
        images, masks = batch
        images, masks = images.to(device), masks.to(device)  # Move data to GPU
        outputs = model(images)
        loss, score = criterion(outputs, masks)
        val_score += score
    merge(to_resolution, save_path=save_path)

if __name__ == "__main__":
    main(path="/lustre/orion/bif146/world-shared/enzhi/miccai_patches", 
        model_weights="/lustre/orion/bif146/world-shared/enzhi/apt/sam-b_miccai-n128-pz8-bz4/",
        patch_size=8,
        resolution=512,
        to_resolution=1024,
        save_path="/lustre/orion/bif146/world-shared/enzhi/miccai_patches/")