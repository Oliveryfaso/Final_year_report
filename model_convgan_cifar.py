# model_convgan_cifar.py

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, num_classes=10, image_size=32):
        super(Generator, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.image_size = image_size

        # 使用embedding来表示标签，而不是one-hot
        # embedding_dim与num_classes相同，即仍为10维，可根据需要调整。
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + num_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf * 8, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 4, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input, labels):
        if input.dim() == 2:
            input = input.view(input.size(0), self.nz, 1, 1)
        elif input.dim() != 4:
            raise ValueError(f"Expected input to have 2 or 4 dimensions, but got {input.dim()}")

        # 使用embedding将label转换为(num_classes,)的向量，然后扩展为 (batch, num_classes, 1, 1)
        embedded_labels = self.label_emb(labels).view(labels.size(0), self.num_classes, 1, 1)

        combined_input = torch.cat([input, embedded_labels], 1)
        output = self.main(combined_input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, num_classes=10, input_size=32):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
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
        real_or_fake = self.fc(features)
        class_output = self.classifier(features)
        return real_or_fake, class_output, features


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('InstanceNorm') != -1 or classname.find('BatchNorm') != -1:
        # 对归一化层权重初始化
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class KwayNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(KwayNetwork, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
