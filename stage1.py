import torch
import torch.nn as nn

import os

import config as cfg

import conditioning_augmentation as ca


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

# up sampling
def upBlock(in_channels, out_channels):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )
    return block


class Stage1_Generator(nn.Module):
    def __init__(self):
        super(Stage1_Generator, self).__init__()
        self.z_dim = cfg.Z_DIM # dim. latentnog prostora
        self.condition_dim = cfg.GAN_CONDITION_DIM
        self.gf_dim = cfg.GAN_GF_DIM

        self.ca_net = ca.CANet()
        
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim + self.condition_dim, self.gf_dim * 8 * 4 * 4),
            nn.BatchNorm1d(self.gf_dim * 8 * 4 * 4),
            nn.ReLU(True)
        )
        
        self.upsample1 = upBlock(self.gf_dim * 8, self.gf_dim * 4)
        self.upsample2 = upBlock(self.gf_dim * 4, self.gf_dim * 2)
        self.upsample3 = upBlock(self.gf_dim * 2, self.gf_dim)
        self.upsample4 = upBlock(self.gf_dim, 3)
        
        self.final = nn.Sequential(
            conv3x3(3, 3),
            nn.Tanh()
        )
    
    def forward(self, text_embedding, noise):
        c, mu, logvar = self.ca_net(text_embedding)

        # spajanje suma i uslovnog vektora
        input = torch.cat((noise, c), 1)

        x = self.fc(input)
        x = x.view(-1, self.gf_dim * 8, 4, 4)

        # upsampling
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)

        x = self.final(x)

        # vracamo generisane slike i statistike za regularizaciju
        return x, mu, logvar 
    


class Stage1_Discriminator(nn.Module):
    def __init__(self):
        super(Stage1_Discriminator, self).__init__()
        self.df_dim = cfg.GAN_DF_DIM
        self.condition_dim = cfg.GAN_CONDITION_DIM

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.df_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv3x3(self.df_dim, self.df_dim * 2),
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            conv3x3(self.df_dim * 2, self.df_dim * 4),
            nn.BatchNorm2d(self.df_dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            conv3x3(self.df_dim * 4, self.df_dim * 8),
            nn.BatchNorm2d(self.df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.condition_fc = nn.Sequential(
            nn.Linear(self.condition_dim, self.df_dim * 8 * 4 * 4),
            nn.ReLU(True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.df_dim * 16, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image, condition):
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Obrada uslovnog vektora
        condition = self.condition_fc(condition)
        condition = condition.view(-1, self.df_dim * 8, 4, 4)

        # Spajanje osobina slike i uslovnog vektora
        x = torch.cat((x, condition), 1)

        x = self.final_conv(x)
        x = x.view(-1, 1)
        return x



