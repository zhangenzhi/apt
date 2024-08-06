import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



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
    
class DiceQDTLoss(nn.Module):
    def __init__(self, weight=0.5, patch_size=8, num_class=2):
        super(DiceQDTLoss, self).__init__()
        self.weight = weight
        self.num_class = num_class
        self.patch_size = patch_size
        self.soft_max = torch.nn.Softmax(-1)

    def forward(self, inputs, targets, qdt_value, smooth=1/256):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        # import pdb
        # pdb.set_trace()
        batch_size = inputs.shape[0]
        fixed_length = inputs.shape[-1]//self.patch_size*inputs.shape[-1]//self.patch_size
        
        inputs = torch.flatten(inputs[:,1:,:,:])
        targets = torch.flatten(targets[:,1:,:,:])
        value = torch.reshape(qdt_value,shape=(batch_size, fixed_length))
        
        pred = torch.reshape(inputs,shape=(fixed_length, -1))
        true = torch.reshape(targets, shape=(fixed_length, -1))
        intersection = torch.sum(pred * true,dim=-1)
        dice_loss = 1 - torch.div(2.*intersection + smooth, torch.sum(pred,dim=-1) + torch.sum(true, dim=-1) + smooth)
        value = torch.sum(value,dim=0)/batch_size
        # value = self.soft_max(value)
        weighted_loss = torch.sum(dice_loss*value)/fixed_length
        
        return weighted_loss

class DiceCLoss(nn.Module):
    def __init__(self, weight=0.5, num_class=2, size_average=True):
        super(DiceCLoss, self).__init__()
        self.weight = weight
        self.num_class = num_class

    def forward(self, inputs, targets, smooth=1e-4, act=True):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if act:
            inputs = F.sigmoid(inputs)       
        
        # pred = torch.flatten(inputs)
        # true = torch.flatten(targets)
        
        # #flatten label and prediction tensors
        pred = torch.flatten(inputs[:,1:,:,:])
        true = torch.flatten(targets[:,1:,:,:])
        
        intersection = (pred * true).sum()
        # coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)/self.num_class                                        
        dice_loss = 1 - (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)  
        dice_bce1 = (1-self.weight)*dice_loss 
        
        pred1 = torch.flatten(inputs[:,0,:,:])
        true1 = torch.flatten(targets[:,0,:,:])
        
        intersection1 = (pred1 * true1).sum()
        # coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)/self.num_class                                        
        dice_loss1 = 1 - (2.*intersection1 + smooth)/(pred1.sum() + true1.sum() + smooth)  
        dice_bce2 = (1-self.weight)*dice_loss1 
        
        return dice_bce1+dice_bce2
    
class DiceBLoss(nn.Module):
    def __init__(self, weight=0.5, num_class=2, size_average=True):
        super(DiceBLoss, self).__init__()
        self.weight = weight
        self.num_class = num_class

    def forward(self, inputs, targets, smooth=1, act=True):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if act:
            inputs = F.sigmoid(inputs)       
        
        # pred = torch.flatten(inputs)
        # true = torch.flatten(targets)
        
        # #flatten label and prediction tensors
        pred = torch.flatten(inputs[:,1:,:,:])
        true = torch.flatten(targets[:,1:,:,:])
        
        intersection = (pred * true).sum()
        coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)                                        
        dice_loss = 1 - (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)  
        BCE = F.binary_cross_entropy(pred, true, reduction='mean')
        dice_bce = self.weight*BCE + (1-self.weight)*dice_loss
        # dice_bce = dice_loss 
        
        return dice_bce
    
    
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
        coeff = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)                                        
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = self.weight*BCE + (1-self.weight)*dice_loss
        
        return Dice_BCE, coeff
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()