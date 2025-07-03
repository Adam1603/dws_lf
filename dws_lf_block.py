# dws_lf_block.py
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.depthwise(x)))
        x = self.relu(self.bn(self.pointwise(x)))
        return x

class DWSLFBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, num_classes=7):
        super(DWSLFBlock, self).__init__()
        
        # Late Fusion (concat features from two branches)
        self.late_fusion = lambda x1, x2: torch.cat((x1, x2), dim=1)
        fused_channels = in_channels1 + in_channels2

        # Stage 1: Conv + DWSConv
        self.conv1 = nn.Conv2d(fused_channels, 64, kernel_size=3, padding=1)
        self.dwsc1 = DepthwiseSeparableConv(fused_channels, 64)

        # Concatenate outputs of conv1 and dwsc1
        self.concat1_channels = 64 + 64

        # Stage 2: Concatenate two DWSConvs
        self.dwsc2a = DepthwiseSeparableConv(self.concat1_channels, 64)
        self.dwsc2b = DepthwiseSeparableConv(self.concat1_channels, 64)
        self.concat2_channels = 64 + 64

        # Stage 3: Conv + Conv + DWSC
        self.conv2 = nn.Conv2d(self.concat2_channels, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dwsc3 = DepthwiseSeparableConv(128, 128)

        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x1, x2):
        # 1. Late Fusion
        x = self.late_fusion(x1, x2)  # [B, C1+C2, H, W]

        # 2. Conv + DWSC
        conv_out = self.conv1(x)
        dwsc_out = self.dwsc1(x)
        x = torch.cat((conv_out, dwsc_out), dim=1)

        # 3. Two DWSConvs, then concatenate
        dwsc2a_out = self.dwsc2a(x)
        dwsc2b_out = self.dwsc2b(x)
        x = torch.cat((dwsc2a_out, dwsc2b_out), dim=1)

        # 4. Conv → Conv → DWSC
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dwsc3(x)

        # 5. Global Pooling + FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
