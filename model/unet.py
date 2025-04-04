import sys
sys.path.append("./")
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import BasicBlock 
from torch.utils.checkpoint import checkpoint  # For gradient checkpointing

# class Decoder(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super(Decoder, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv_relu = nn.Sequential(
#             nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x1 = torch.cat((x1, x2), dim=1)
#         x1 = self.conv_relu(x1)
#         return x1

# class Unet(nn.Module):
#     def __init__(self, n_class, in_channels=3, pretrained=True):
#         super().__init__()
#         # self.up_first = nn.ConvTranspose2d(in_channels=in_channels, out_channels=3, kernel_size=2, stride=2)
#         self.base_model = torchvision.models.resnet18()
#         self.base_model.load_state_dict(torch.load("./model/resnet18-f37072fd.pth"))
#         for param in self.base_model.parameters():
#             param.requires_grad = False  # freeze
#         self.base_layers = list(self.base_model.children())
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             self.base_layers[1],
#             self.base_layers[2])
#         self.layer2 = nn.Sequential(*self.base_layers[3:5])
#         self.layer3 = self.base_layers[5]
#         self.layer4 = self.base_layers[6]
#         self.layer5 = self.base_layers[7]
#         self.decode4 = Decoder(512, 128+256, 128)
#         self.decode3 = Decoder(128, 128+128, 128)
#         self.decode2 = Decoder(128, 64+64, 64)
#         self.decode1 = Decoder(64, 16+64, 16)
#         self.conv_last = nn.Conv2d(16, n_class, 1, 1)
#         self.upscale = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#         )

#     def forward(self, input):
#         # input = self.up_first(input)
        
#         e1 = self.layer1(input)  # 64,16,385*16
#         e2 = self.layer2(e1)     # 64,8,385*8
#         e3 = self.layer3(e2)     # 128,4,385*4
#         e4 = self.layer4(e3)     # 256,2,385*2
#         f = self.layer5(e4)      # 512,1,385*1
#         d4 = self.decode4(f, e4) # 256,32,32
#         d3 = self.decode3(d4, e3) # 256,64,64
#         d2 = self.decode2(d3, e2) # 128,128,128
#         d1 = self.decode1(d2, e1) # 64,256,256
#         # d0 = self.decode0(d1)     # 64,512,512
#         out = self.conv_last(d1)  # 1,256,256
#         out = self.upscale(out)
        
#         return out


class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),  # Reduced group count
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential size mismatches
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, n_class=5, in_channels=1, pretrained=True):
        super().__init__()
        
        # Modified ResNet18 backbone with reduced channels
        base_model = torchvision.models.resnet18(pretrained=False)
        base_model.load_state_dict(torch.load("./model/resnet18-f37072fd.pth"))
        
        # Halve all channel counts
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),  # 64->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        
        # Modify ResNet layers to reduce channels
        self.encoder2 = self._make_res_layer(base_model.layer1, 32)   # Original: 64
        self.encoder3 = self._make_res_layer(base_model.layer2, 64)   # Original: 128
        self.encoder4 = self._make_res_layer(base_model.layer3, 96)   # Original: 256
        self.encoder5 = self._make_res_layer(base_model.layer4, 128)  # Original: 512
        
        # Decoder with proportional reduction
        self.decoder4 = Decoder(128, 96, 96)    # Original: 512,256,256
        self.decoder3 = Decoder(96, 64, 64)     # Original: 256,128,128
        self.decoder2 = Decoder(64, 32, 32)     # Original: 128,64,64
        self.decoder1 = Decoder(32, 32, 16)     # Original: 64,64,32
        
        # Final layers
        self.final_conv = nn.Conv2d(16, n_class, kernel_size=1)
        
        # Simplified upsampling
        self.mega_upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def _make_res_layer(self, original_layer, out_channels):
        """Modify a ResNet layer to have reduced output channels"""
        layers = []
        for i, block in enumerate(original_layer.children()):
            # Create a copy with reduced channels
            new_block = BasicBlock(
                block.conv1.in_channels if i==0 else out_channels,
                out_channels,
                stride=block.stride,
                downsample=block.downsample
            )
            layers.append(new_block)
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)     # [1, 32, 4096, 4096]
        e2 = self.encoder2(e1)    # [1, 32, 4096, 4096]
        e3 = self.encoder3(e2)    # [1, 64, 2048, 2048]
        e4 = self.encoder4(e3)    # [1, 96, 1024, 1024]
        f = self.encoder5(e4)     # [1, 128, 512, 512]
        
        # Decoder
        d4 = self.decoder4(f, e4)     # [1, 96, 1024, 1024]
        d3 = self.decoder3(d4, e3)    # [1, 64, 2048, 2048]
        d2 = self.decoder2(d3, e2)    # [1, 32, 4096, 4096]
        d1 = self.decoder1(d2, e1)    # [1, 16, 8192, 8192]
        
        out = self.final_conv(d1)     # [1, 5, 8192, 8192]
        
        if out.size()[-2:] != x.size()[-2:]:
            out = self.mega_upsample(out)
        return out


class LightDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),  # Replaces BatchNorm for stability with small batches
            nn.ReLU(inplace=True)
        )
    def _custom_checkpoint(self, module, input):
        input = input.detach()
        input.requires_grad_(True)
        output = module(input)
        return output

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatches (critical for high-res)
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self._custom_checkpoint(self.conv, x)  # Gradient checkpointing saves memory

class LightweightUNet(nn.Module):
    def __init__(self, n_class=5, in_channels=1):
        super().__init__()
        
        # Encoder (fewer channels in deeper layers)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # Downsample to 4096x4096
        )
        self.encoder2 = self._make_layer(32, 64, stride=2)   # 2048x2048
        self.encoder3 = self._make_layer(64, 96, stride=2)   # 1024x1024  [Reduced from 128]
        self.encoder4 = self._make_layer(96, 128, stride=2)  # 512x512    [Reduced from 256]
        
        # Decoder (matching reduced channels)
        self.decoder3 = LightDecoder(128, 96, 96)    # 1024x1024
        self.decoder2 = LightDecoder(96, 64, 64)     # 2048x2048
        self.decoder1 = LightDecoder(64, 32, 32)     # 4096x4096
        
        # Final upsampling and output
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, n_class, kernel_size=1)
        )

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def _custom_checkpoint(self, module, input):
        input = input.detach()
        input.requires_grad_(True)
        output = module(input)
        return output
    
    def forward(self, x):
        # Encoder (with gradient checkpointing)
        e1 = self._custom_checkpoint(self.encoder1, x)      # [1, 32, 4096, 4096]
        e2 = self._custom_checkpoint(self.encoder2, e1)     # [1, 64, 2048, 2048]
        e3 = self._custom_checkpoint(self.encoder3, e2)     # [1, 96, 1024, 1024]
        e4 = self._custom_checkpoint(self.encoder4, e3)     # [1, 128, 512, 512]
        
        # Decoder
        d3 = self.decoder3(e4, e3)            # [1, 96, 1024, 1024]
        d2 = self.decoder2(d3, e2)            # [1, 64, 2048, 2048]
        d1 = self.decoder1(d2, e1)             # [1, 32, 4096, 4096]
        
        # Final upsampling to original resolution
        out = self.final_upsample(d1)          # [1, 5, 8192, 8192]
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