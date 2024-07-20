import numpy as np
import cv2 as cv
import torch
import random
from apt.quadtree import FixedQuadTree

class Patchify(torch.nn.Module):
    def __init__(self, sths=[1,3,5,7], fixed_length=1024, cannys=[50, 150], patch_size=8) -> None:
        super().__init__()
        
        self.sths = sths
        self.fixed_length = fixed_length
        self.cannys = [x for x in range(cannys[0], cannys[1], 1)]
        self.patch_size = patch_size
        
    def forward(self, img, target):  # we assume inputs are always structured like this
        # Do some transformations. Here, we're just passing though the input
        
        self.smooth_factor = random.choice(self.sths)
        c = random.choice(self.cannys)
        self.canny = [c, c+30]
        
        grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
        edges = cv.Canny(grey_img, self.canny[0], self.canny[1])
        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        seq_img = qdt.serialize(img, size=(self.patch_size,self.patch_size,3))
        seq_img = np.asarray(seq_img)
        seq_img = np.reshape(seq_img, [self.patch_size, -1, 3])
        
        seq_mask = qdt.serialize(target, size=(self.patch_size, self.patch_size, 1))
        seq_mask = np.asarray(seq_mask)
        seq_mask = np.reshape(seq_mask, [self.patch_size, -1, 1])

        return seq_img, seq_mask, qdt