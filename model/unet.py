import sys
sys.path.append("./")
import torch
import torchvision
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights

class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1

class Unet(nn.Module):
    def __init__(self, n_class, in_channels=3, pretrain=True):
        super().__init__()
        # self.up_first = nn.ConvTranspose2d(in_channels=in_channels, out_channels=3, kernel_size=2, stride=2)
        self.base_model = torchvision.models.resnet18()
        self.base_model.load_state_dict(torch.load("./model/resnet18-f37072fd.pth"))
        
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 16+64, 16)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            # nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            # nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        )
        self.conv_last = nn.Conv2d(16, n_class, 1, 2)

    def forward(self, input):
        # input = self.up_first(input)
        
        e1 = self.layer1(input)  # 64,16,385*16
        e2 = self.layer2(e1)     # 64,8,385*8
        e3 = self.layer3(e2)     # 128,4,385*4
        e4 = self.layer4(e3)     # 256,2,385*2
        f = self.layer5(e4)      # 512,1,385*1
        d4 = self.decode4(f, e4) # 256,32,32
        d3 = self.decode3(d4, e3) # 256,64,64
        d2 = self.decode2(d3, e2) # 128,128,128
        d1 = self.decode1(d2, e1) # 64,256,256
        input = self.decode0(d1)     # 64,512,512
        out = self.conv_last(input)  # 1,512,512
        
        import pdb;pdb.set_trace()
        
        return out
    
if __name__ == '__main__':
    import torch

    Unet = Unet(n_class=1)

    print(sum(p.numel() for p in Unet.parameters()))
    print(Unet(torch.randn(1, 3, 16384, 16384)).shape)
    
    from calflops import calculate_flops
    batch_size = 1
    input_shape = (batch_size, 3, 16384, 16384)
    flops, macs, params = calculate_flops(model=Unet, 
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4)
    print("Unetr FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))