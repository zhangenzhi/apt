import numpy as np
import torch
import cv2 as cv
from matplotlib import pyplot as plt

class Rect:
    def __init__(self, x1, x2, y1, y2) -> None:
        # *q
        # p*
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
        assert x1<=x2, 'x1 > x2, wrong coordinate.'
        assert y1<=y2, 'y1 > y2, wrong coordinate.'
    
    def contains(self, domain):
        patch = domain[self.y1:self.y2, self.x1:self.x2]
        return int(np.sum(patch)/255)
    
    def get_area(self, img):
        return img[self.y1:self.y2, self.x1:self.x2, :]
    
    def set_area(self, mask, patch):
        patch_size = self.get_size()
        patch = cv.resize(patch, interpolation=cv.INTER_CUBIC , dsize=patch_size)
        mask[self.y1:self.y2, self.x1:self.x2, :] = patch
        return mask
    
    def get_coord(self):
        return self.x1,self.x2,self.y1,self.y2
    
    def get_size(self):
        return self.x2-self.x1, self.y2-self.y1
    
    def draw(self, ax, c='grey', lw=0.5, **kwargs):
        # Create a Rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x1, self.y1), 
                                 width=self.x2-self.x1, 
                                 height=self.y2-self.y1, 
                                 linewidth=lw, edgecolor='w', facecolor='none')
        ax.add_patch(rect)
    
                 
class FixedQuadTree:
    def __init__(self, domain, fixed_length=128) -> None:
        self.domain = domain
        self.fixed_length = fixed_length
        self._build_tree()
        
    def _build_tree(self):
    
        h,w = self.domain.shape
        assert h>0 and w >0, "Wrong img size."
        root = Rect(0,w,0,h)
        self.nodes = [(root, root.contains(self.domain))]
        while len(self.nodes)<self.fixed_length:
            bbox, value = max(self.nodes, key=lambda x:x[1])
            idx = self.nodes.index((bbox, value))
        
            x1,x2,y1,y2 = bbox.get_coord()
            lt = Rect(x1, int((x1+x2)/2), int((y1+y2)/2), y2)
            v1 = lt.contains(self.domain)
            rt = Rect(int((x1+x2)/2), x2, int((y1+y2)/2), y2)
            v2 = rt.contains(self.domain)
            lb = Rect(x1, int((x1+x2)/2), y1, int((y1+y2)/2))
            v3 = lb.contains(self.domain)
            rb = Rect(int((x1+x2)/2), x2, y1, int((y1+y2)/2))
            v4 = rb.contains(self.domain)
            
            self.nodes = self.nodes[:idx] + [(lt,v1), (rt,v2), (lb,v3), (rb,v4)] +  self.nodes[idx+1:]

            # print([v for _,v in self.nodes])
            
    def count_patches(self):
        return len(self.nodes)
    
    def serialize(self, img, size=(8,8,3)):
        
        seq_patch = []
        for bbox,value in self.nodes:
            seq_patch.append(bbox.get_area(img))
            
        h2,w2,c2 = size
        for i in range(len(seq_patch)):
            h1, w1, c1 = seq_patch[i].shape
            # assert h1==w1, "Need squared input."
            seq_patch[i] = cv.resize(seq_patch[i], (h2, w2), interpolation=cv.INTER_CUBIC)
            assert seq_patch[i].shape == (h2,w2,c2), "Wrong shape {} get, need {}".format(seq_patch[i].shape, (h2,w2,c2))
            
        return seq_patch
    
    def deserialize(self, seq, mask):
        for bbox,value in self.nodes:
            pred_mask = seq.pop(0)
            mask = bbox.set_area(mask, pred_mask)
        return mask
    
    def draw(self, ax, c='grey', lw=1):
        for bbox,value in self.nodes:
            bbox.draw(ax=ax)
    
                
