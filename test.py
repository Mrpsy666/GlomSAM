# -*- coding: utf-8 -*-
# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import albumentations as A
join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from datetime import datetime
import shutil
import monai
from utils import (
    BoundaryDoULoss, 
    UNet,
    check_accuracy,
    plot_accuracy
)
from MyDataset import MyDataset
import random

# set seeds and 释放显存
torch.manual_seed(2023)
torch.cuda.empty_cache()

test_img = '../data/class3/10/test/images'
test_mask = '../data/class3/10/test/masks'
# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument("-task_name", type=str, default="GlomSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="./weights/20x/GlomSAM.pth"
)

# train
parser.add_argument("-num_epochs", type=int, default=200)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=1)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

device = torch.device(args.device)
# %% set up model

class GlomSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        RougMaskGenerator,
    ) -> None:
        super().__init__()
        self.image_encoder=image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        self.RoughMaskGenerator = RougMaskGenerator
        self.RoughMaskGenerator.eval()

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = True
        for param in self.mask_decoder.parameters():
            param.requires_grad = True


        if hasattr(self.image_encoder, 'prompt_generator'):
            for param in self.image_encoder.prompt_generator.parameters():
                param.requires_grad = True
        
        if hasattr(self.image_encoder,'cnn_branch'):
            for param in self.image_encoder.cnn_branch.parameters():
                param.requires_grad = True
        
        if hasattr(self.image_encoder,'fusion_module'):
            for param in self.image_encoder.fusion_module.parameters():
                param.requires_grad = True

    def forward(self,image,box):
        RoughMask = F.interpolate(
                self.RoughMaskGenerator(
                    F.interpolate(
                        image,
                        size=(512,512),
                        mode="bilinear",
                        align_corners=False
                    )
                ),
                    size=(256,256),
                    mode="bilinear",
                    align_corners=False
        )
        

        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box,
            masks=RoughMask,
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def main():
    RoughMaskGenerator= UNet().to(device)
    RoughMaskGeneratorcheckpoint = torch.load("./weights/10x/RoughMaskGenerator", map_location=device)
    RoughMaskGenerator.load_state_dict(RoughMaskGeneratorcheckpoint["state_dict"])
    RoughMaskGenerator.eval()

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    glomsam_model = GlomSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        RoughMaskGenerator=RoughMaskGenerator
    ).to(device)
    glomsam_model.eval()


    image_transform = A.Compose(
        [
            A.Resize(height=1024,width=1024,interpolation=cv2.INTER_AREA),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
        ],
    )
    test_dataset = MyDataset(test_img, test_mask,transform=image_transform)


    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )



    epoch_score = check_accuracy(test_dataloader,glomsam_model,args.device)

if __name__ == "__main__":
    main()