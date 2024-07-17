import torch

class MyCustomTransform(torch.nn.Module):
    def __init__(self, smooth_factor=1, fixed_length=1024, canny=[100, 200]) -> None:
        super().__init__()
        self.smooth_factor = smooth_factor
        self.fixed_length = fixed_length
        self.canny = canny
    def forward(self, img, target):  # we assume inputs are always structured like this
        # Do some transformations. Here, we're just passing though the input
        
        return img, target