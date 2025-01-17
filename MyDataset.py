# -*- coding: utf-8 -*-
import os 
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import re
import torch.nn.functional as F

import torch.nn.functional as F
import torch.nn as nn

class MyDataset(Dataset):
    def __init__(self,image_dir,mask_dir, transform=None,bbox_shift=20):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.bbox_shift = bbox_shift

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):


        img_path = os.path.join(self.image_dir,self.images[index])
        mask_path = os.path.join(self.mask_dir,self.images[index])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"),dtype=np.float32)
        mask[mask == 255.0] = 1.0

        #在transform中将totensor省略
        if self.transform is not None:
            augmentation = self.transform(image=image,mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]
        

        image = np.transpose(image, (2, 0, 1))
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = mask.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(image).float(),
            torch.tensor(mask[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            self.images[index]
        )