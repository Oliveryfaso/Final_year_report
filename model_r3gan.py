import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F  # 新增导入

##############################################################################
# 1. 生成器用残差块 (若想去掉BN，可按判别器的写法删除BN)
##############################################################################
class ResBlockG(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False))
        self.bn1   = nn.BatchNorm2d(out_ch, affine=True)
        self.act1  = nn.ReLU(inplace=True)

        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False))
        self.bn2   = nn.BatchNorm2d(out_ch, affine=True)
        self.act2  = nn.ReLU(inplace=True)

        # shortcut 分支
        self.shortcut = nn.Identity()
        if downsample or (in_ch != out_ch):
            self.shortcut = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        if isinstance(self.shortcut, nn.Conv2d):
            nn.init.kaiming_normal_(self.shortcut.weight, mode='fan_in', nonlinearity='relu')

        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn1.bias,   0.0)
        nn.init.constant_(self.bn2.weight, 1.0)
        nn.init.constant_(self.bn2.bias,   0.0)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act2(out)
        return out

##############################################################################
# 2. 判别器用残差块：彻底去掉BN，仅用SpectralNorm+LeakyReLU (在WGAN/R3GAN中更稳定)
##############################################################################
class ResBlockD(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False))
        self.act1  = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False))
        self.act2  = nn.LeakyReLU(0.2, inplace=True)

        # shortcut
        self.shortcut = nn.Identity()
        if downsample or (in_ch != out_ch):
            self.shortcut = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False))

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = out + identity
        out = self.act2(out)
        return out

##############################################################################
# 3. 判别器 R3Discriminator
#    - 首层卷积 + LeakyReLU
#    - 若干个 ResBlockD
#    - 全局池化 + 最终线性输出
##############################################################################
class R3Discriminator(nn.Module):
    def __init__(self, img_ch=3, ndf=64):
        super().__init__()
        # 特征提取模块
        self.blocks = nn.Sequential(
            spectral_norm(nn.Conv2d(img_ch, ndf, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            ResBlockD(ndf, ndf*2, downsample=True),   # 32x32 → 16x16
            ResBlockD(ndf*2, ndf*4, downsample=True), # 16x16 → 8x8
            ResBlockD(ndf*4, ndf*8, downsample=True), # 8x8 → 4x4
            
            nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        )
        
        self.fc = nn.Sequential(
            nn.Linear(ndf*8, 512),  # 新增中间层
            nn.LayerNorm(512),       # 层归一化
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(512, 1))  # 最终输出层
        )
        
        # 初始化修正
        nn.init.kaiming_normal_(self.fc[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(self.fc[0].bias, 0.0)
        nn.init.orthogonal_(self.fc[3].weight)

    def forward(self, x):
        """判别器前向传播流程"""
        # 特征提取
        features = self.blocks(x)  # [B, ndf*8, 1, 1]
        
        # 分类头
        features = features.view(features.size(0), -1)  # [B, ndf*8]
        validity = self.fc(features)  # [B, 1]
        return validity

class R3Generator(nn.Module):
    def __init__(self, nz=128, ngf=64, img_ch=3):
        super().__init__()
        self.init_size = 4
        # 初始化层（增加稳定性设计）
        self.l1 = nn.Sequential(
            nn.Linear(nz, ngf*8*self.init_size**2),
            nn.BatchNorm1d(ngf*8*self.init_size**2),  # 添加BN层
            nn.LeakyReLU(0.2, inplace=True)
        )
        nn.init.kaiming_normal_(self.l1[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(self.l1[0].bias, 0.0)
        
        # 生成器主体（保持原有结构）
        self.blocks = nn.Sequential(
            ResBlockG(ngf*8, ngf*8),
            ResBlockG(ngf*8, ngf*4),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 改用最近邻上采样
            ResBlockG(ngf*4, ngf*2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlockG(ngf*2, ngf),
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(ngf, img_ch, 3, 1, 1)),
            nn.Tanh()
        )

    def forward(self, z):
        # 阶段1：潜在空间映射
        out = self.l1(z)  # [B, nz] → [B, ngf*8 * 4 * 4]
        out = out.view(-1, self.blocks[0].conv1.in_channels, 
                      self.init_size, self.init_size)  # 重塑为4x4特征图
        
        # 阶段2：多尺度特征生成
        for layer in self.blocks:
            # 残差块与上采样的交替处理
            if isinstance(layer, nn.Upsample):
                out = layer(out)
                out = F.leaky_relu(out, 0.2)  # 上采样后激活
            else:
                out = layer(out)
        
        return out