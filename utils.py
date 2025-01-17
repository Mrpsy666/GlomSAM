import torch
import torch.nn.functional as F
import torch.nn as nn

import os
join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
           nn.Conv2d(in_channels,out_channels,3,1,1,bias=True),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace=True),
           nn.Conv2d(out_channels,out_channels,3,1,1,bias=True),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,features=[64,128,256,512],
    ):
        super(UNet,self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        #### Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature
        
        #### Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                feature*2,feature,kernel_size=2,stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2,feature))

        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):
        skip_connections=[]
        

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.resize(x,size=skip_connection.shape[2:],mode='bilinear',align_corners=True)
        
            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


###### nested unet 3channels are required
class conv_block_nested(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super(conv_block_nested,self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output
class BoundaryDoULoss(nn.Module):
    def __init__(self):
        super(BoundaryDoULoss, self).__init__()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        kernel = kernel.float()
        padding_out = torch.zeros((target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2))
        padding_out[:, 1:-1, 1:-1] = target.squeeze(1)
        h, w = 3, 3
        Y = torch.zeros((padding_out.shape[0],padding_out.shape[1]-h+1,padding_out.shape[2]-w+1)).cuda()
        for i in range(Y.shape[0]):
            Y[i,:,:] = torch.conv2d(target[i].unsqueeze(0).cuda(), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        # Y = torch.conv2d(target.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y.to("cuda:1") * target.to("cuda:1")
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        target = target.float()  # Assuming target is a tensor of 0s and 1s
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = self._adaptive_size(inputs, target)
        return loss
    

def check_accuracy(loader, glomsam_model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0.0
    iou = 0.0
    recall = 0.0  # 添加Recall总量
    glomsam_model.eval()

    # 创建用于保存ground truth和预测结果的文件夹
    os.makedirs("gts", exist_ok=True)
    os.makedirs("preds", exist_ok=True)

    with torch.no_grad():
        for step, (img, mask, boxes, img_names) in enumerate(tqdm(loader)):
            image, mask, boxes = img.to(device), mask.to(device), boxes.to(device)
            
            preds = glomsam_model(image, boxes)
            preds = (preds > 0.5).long()
            num_correct += (preds == mask).sum().item()
            num_pixels += torch.numel(preds)

            intersection = torch.logical_and(preds, mask).sum().item()
            union = torch.logical_or(preds, mask).sum().item()
            iou += intersection / union if union > 0 else 0.0
            dice_score += (2 * intersection) / (
                (preds.sum().item() + mask.sum().item()) + 1e-8
            )

            # 计算Recall
            true_positive = intersection
            false_negative = (mask * (preds == 0)).sum().item()
            recall += true_positive / (true_positive + false_negative + 1e-8)

            # 保存ground truth和预测的掩码
            for i in range(image.size(0)):
                pred_mask_np = preds[i].cpu().numpy().astype(np.uint8) * 255
                target_mask_np = mask[i].cpu().numpy().astype(np.uint8) * 255

                pred_mask_img = Image.fromarray(pred_mask_np.squeeze(), mode='L')
                target_mask_img = Image.fromarray(target_mask_np.squeeze(), mode='L')

                # 使用 img_names[i] 以确保保存单个文件的名称
                pred_mask_img.save(os.path.join("preds", img_names[i]))
                target_mask_img.save(os.path.join("gts", img_names[i]))

    avg_accuracy = num_correct / num_pixels * 100
    avg_iou = iou / len(loader)
    avg_dice = dice_score / len(loader)
    avg_recall = recall / len(loader)  # 计算平均Recall

    print(f"Got {num_correct}/{num_pixels} with acc {avg_accuracy:.8f}%")
    print(f"IoU score :{avg_iou:.8f}")
    print(f"Dice score: {avg_dice:.8f}")
    print(f"Recall score: {avg_recall:.8f}")  # 打印平均Recall

    return avg_dice



import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

def plot_accuracy(loader, glomsam_model, device):
    num_correct = 0
    num_pixels = 0
    dice_scores = []
    iou_scores = []
    accuracies = []
    glomsam_model.eval()

    with torch.no_grad():
        for step, (img, mask, boxes) in enumerate(tqdm(loader)):
            boxes_np = boxes.detach().cpu().numpy()
            image, mask = img.to(device), mask.to(device)

            preds = glomsam_model(image, boxes_np)
            preds = (preds > 0.5).float()
            num_correct += (preds == mask).sum().item()
            num_pixels += torch.numel(preds)

            intersection = torch.logical_and(preds, mask).sum().item()
            union = torch.logical_or(preds, mask).sum().item()
            iou_score = intersection / union if union > 0 else 0.0
            dice_score = (2 * intersection) / (preds.sum().item() + mask.sum().item() + 1e-8)
            accuracy = num_correct / num_pixels

            iou_scores.append(iou_score)
            dice_scores.append(dice_score)
            accuracies.append(accuracy)

    print("dice:")
    print(dice_scores)
    print("IOU:")
    print(iou_scores)
    print("pixcel:")
    print(accuracies)

    print(f"Average IoU score: {sum(iou_scores) / len(loader):.4f}")
    print(f"Average Dice score: {sum(dice_scores) / len(loader):.4f}")
    print(f"Average accuracy: {sum(accuracies) / len(loader):.2f}%")

    return sum(dice_scores) / len(loader)



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for Binary Classification Tasks.
        
        Parameters:
            alpha (float): Weighting factor for the rare class (usually < 1).
            gamma (float): Modulating factor to focus on hard examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for the focal loss calculation.
        
        Parameters:
            inputs (tensor): Predicted probabilities for the positive class.
            targets (tensor): Ground truth labels, where 1 represents the positive class and 0 the negative class.
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.float32)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
