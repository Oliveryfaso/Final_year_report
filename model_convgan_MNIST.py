# model_convgan_MNIST.py

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, num_classes=24):
        super(Generator, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + num_classes, ngf * 4, 3, 1, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        )

    def forward(self, input, labels):
        if input.dim() == 2:
            input = input.view(input.size(0), self.nz, 1, 1)
        elif input.dim() != 4:
            raise ValueError(f"Expected input to have 2 or 4 dimensions, but got {input.dim()}")

        labels_onehot = torch.zeros(input.size(0), self.num_classes, 1, 1, device=input.device)
        labels_onehot.scatter_(1, labels.view(-1, 1, 1, 1), 1)

        combined_input = torch.cat([input, labels_onehot], 1)
        output = self.main(combined_input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64, num_classes=24, input_size=28):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, nc, input_size, input_size)
            dummy_output = self.features(dummy_input)
            self.flatten_size = dummy_output.numel() // dummy_output.size(0)

        self.fc = spectral_norm(nn.Linear(self.flatten_size, 1))
        self.classifier = spectral_norm(nn.Linear(self.flatten_size, num_classes))

    def forward(self, input):
        features = self.features(input)
        features = features.view(features.size(0), -1)
        fc_output = self.fc(features)
        real_or_fake = fc_output
        class_output = self.classifier(features)
        return real_or_fake, class_output, features  # 返回中间层特征

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class KwayNetwork(nn.Module):
    def __init__(self, num_classes=24):
        super(KwayNetwork, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()  # 输出特征
        self.classifier = nn.Linear(num_ftrs, num_classes)
        self.num_classes = num_classes  # 确保有这一行

    def forward(self, x):
        features = self.model(x)
        logits = self.classifier(features)
        return features, logits

def get_optimizer(model, lr=0.0002, beta1=0.5, beta2=0.999):
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))





import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Generator_cWGAN_GP(nn.Module):
    def __init__(self, nz=100, num_classes=24, ngf=64, nc=1):
        super().__init__()
        self.num_classes = num_classes
        self.main = nn.Sequential(
            # 初始层保持不变
            nn.ConvTranspose2d(nz + num_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # 中间层调整
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),  # 4x4 → 7x7
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, output_padding=1, bias=False),  # 7x7 →14x14
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # 最终层使用kernel_size=3调整输出尺寸
            nn.ConvTranspose2d(ngf * 2, nc, 3, 2, 1, output_padding=1, bias=False),  #14x14→28x28 
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # 标签转one-hot编码并与噪声拼接（网页1中标签处理方法）
        batch_size = noise.size(0)
        labels_onehot = torch.zeros(batch_size, self.num_classes, device=noise.device)
        labels_onehot.scatter_(1, labels.view(-1,1), 1)
        
        # 合并噪声和标签（网页1中的条件输入方法）
        combined = torch.cat([noise, labels_onehot], dim=1)
        combined = combined.unsqueeze(2).unsqueeze(3)  # [b,124] → [b,124,1,1]
        
        img = self.main(combined)
        assert img.shape[2:] == (28, 28), f"生成器输出尺寸错误: {img.shape}"
        return img

class Critic_cWGAN_GP(nn.Module):
    def __init__(self, num_classes=24, ndf=64, nc=1, image_size=28):
        super(Critic_cWGAN_GP, self).__init__()
        self.num_classes = num_classes
        
        # 修正卷积层结构确保输出维度匹配
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(nc + num_classes, ndf, 4, 2, 1)),  # 28x28 →14x14
            nn.LayerNorm([ndf, 14, 14]),  # 添加层归一化
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1)),             # 14x14 →7x7
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 3, 2, 1)),            # 7x7 →4x4
            nn.LeakyReLU(0.2),
        )
        
        # 动态计算全连接层输入维度
        with torch.no_grad():
            dummy = torch.zeros(1, nc + num_classes, image_size, image_size)
            dummy_out = self.conv(dummy)
            self.feat_size = dummy_out.view(dummy_out.size(0), -1).size(1)
        
        self.linear = spectral_norm(nn.Linear(self.feat_size, 1))

    def forward(self, x, labels):
        batch_size = x.size(0)
        labels_map = torch.zeros(batch_size, self.num_classes, x.size(2), x.size(3), device=x.device)
        labels_map.scatter_(1, labels.view(-1, 1, 1, 1).expand(-1, -1, x.size(2), x.size(3)), 1)
        x = torch.cat([x, labels_map], dim=1)
        features = self.conv(x).view(batch_size, -1)
        return self.linear(features)