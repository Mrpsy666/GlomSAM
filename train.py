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
)
from MyDataset import MyDataset
import random

# set seeds and 释放显存
torch.manual_seed(2023)
torch.cuda.empty_cache()
train_img = '../data/class3/10/train/images'
train_mask = '../data/class3/10/train/masks'

valid_img = '../data/class3/10/valid/images'
valid_mask = '../data/class3/10/valid/masks'
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
    "-checkpoint", type=str, default="./weights/10x/GlomSAM.pth"
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
parser.add_argument("--device", type=str, default="cuda:1")
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
    glomsam_model.train()


    print(
        "Number of total parameters: ",
        sum(p.numel() for p in glomsam_model.parameters()),
    )  
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in glomsam_model.parameters() if p.requires_grad),
    )
    optimizer = torch.optim.AdamW(
        glomsam_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of params to train: ",
        sum(p.numel() for p in glomsam_model.parameters() if p.requires_grad),
    )  
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    loss_fn = BoundaryDoULoss()

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

    num_epochs = args.num_epochs

    best_loss = 100
    train_dataset = MyDataset(train_img,train_mask,transform=image_transform)
    valid_dataset = MyDataset(valid_img, valid_mask,transform=image_transform)

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
            
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            image, gt2D, boxes = image.to(device), gt2D.to(device),boxes.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):    
                glomsam_pred = glomsam_model(image, boxes)
                #BoundaryDouLOss + DiceLoss+ BCELoss
                loss = 0.5 * ce_loss(glomsam_pred,gt2D.float()) + 0.5 * seg_loss(glomsam_pred,gt2D) + 0.5 * loss_fn(glomsam_pred,gt2D)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss+=loss.item()
        epoch_loss /= step

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')

        if  epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": glomsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join("./output/glomsamcnn_model_best.pth"))
        

if __name__ == "__main__":
    main()