
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet101_Weights

class ResNetCNNBranch(nn.Module):
    def __init__(self, out_chans: int = 256):
        super().__init__()
        # 使用预训练权重加载ResNet101
        weights = ResNet101_Weights.DEFAULT  # 或 ResNet101_Weights.IMAGENET1K_V1 根据需要选择
        self.resnet101 = models.resnet101(weights=weights)
        # 移除最后两层以保持特征提取层
        self.features = nn.Sequential(*list(self.resnet101.children())[:-2])
        # 添加一个适配层以调整输出通道数
        self.adapt_conv = nn.Conv2d(2048, out_chans, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adapt_conv(x)
        return x



class SEBlock(nn.Module):
    def __init__(self, in_chans, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_chans, in_chans // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_chans // reduction, in_chans, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y


class FusionWithSE(nn.Module):
    def __init__(self, vit_chans, cnn_chans, out_chans):
        super(FusionWithSE, self).__init__()
        self.se_block = SEBlock(vit_chans + cnn_chans)
        self.conv = nn.Conv2d(vit_chans + cnn_chans, out_chans, kernel_size=1)

    def forward(self, vit_features, cnn_features):
        # Step 1: Resize cnn_features to match the size of vit_features
        if vit_features.size()[2:] != cnn_features.size()[2:]:
            cnn_features = F.interpolate(cnn_features, size=vit_features.shape[2:], mode='bilinear', align_corners=False)

        # Step 2: Concatenate along the channel dimension
        combined_features = torch.cat((vit_features, cnn_features), dim=1)
        weighted_features = self.se_block(combined_features)
        return self.conv(weighted_features)

