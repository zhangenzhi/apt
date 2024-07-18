import numpy as np
import cv2 as cv
import torch
from apt.quadtree import FixedQuadTree

class Patchify(torch.nn.Module):
    def __init__(self, smooth_factor=1, fixed_length=1024, canny=[100, 200], patch_size=8) -> None:
        super().__init__()
        self.smooth_factor = smooth_factor
        self.fixed_length = fixed_length
        self.canny = canny
        self.patch_size = patch_size
        
    def forward(self, img, target):  # we assume inputs are always structured like this
        # Do some transformations. Here, we're just passing though the input
        grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
        edges = cv.Canny(grey_img, self.canny[0], self.canny[1])
        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        seq_img = qdt.serialize(img, size=(self.patch_size,self.patch_size,3))
        seq_mask = qdt.serialize(target, size=(self.patch_size,self.patch_size,3))
        
        return seq_img, seq_mask, qdt