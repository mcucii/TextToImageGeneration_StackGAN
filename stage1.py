import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

from torch.optim import Adam

import config as cfg
import conditioning_augmentation as ca
import utils


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
            nn.ReLU()
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



class GANTrainer_stage1():
    def __init__(self, output_dir):
        self.model_dir = os.path.join(cfg.DATA_DIR, 'Model')
        self.image_dir = os.path.join(cfg.DATA_DIR, 'Image')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.max_epoch = cfg.TRAIN_MAX_EPOCH
        self.batch_size = cfg.TRAIN_BATCH_SIZE
    
    # def get_imgs():
    #     imgs = []
    #     for i in range(0, 2911):
    #         img_path = "../data/birds/models/netG_epoch_360/" + str(i) + ".png"
    #         img = Image.open(img_path).convert('RGB')
    #         img = np.array(img)
    #         imgs.append(img)
    #     return imgs

    def load_networks(self):
        netG = Stage1_Generator()  
        netG.apply(utils.weight_initialization)

        netD = Stage1_Discriminator()
        netD.apply(utils.weight_initialization)

        return netG, netD


    def train(self, dataloader):
        netG, netD = self.load_networks()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        netG = netG.to(device)
        netD = netD.to(device)
    
        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = torch.FloatTensor(batch_size, nz).to(device) 
        fixed_noise = torch.FloatTensor(batch_size, nz).normal_(0, 1)
        real_labels = torch.FloatTensor(batch_size).fill_(1)
        fake_labels = torch.FloatTensor(batch_size).fill_(0)

        generator_lr = cfg.TRAIN_GENERATOR_LR
        discriminator_lr = cfg.TRAIN_DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN_LR_DECAY_EPOCH

        optimizerG = Adam(netG.parameters(), lr=cfg.TRAIN_GENERATOR_LR)
        optimizerD = Adam(netD.parameters(), lr=cfg.TRAIN_DISCRIMINATOR_LR)
        
        for epoch in range(self.max_epoch):
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr

                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr            

            for i, data in enumerate(dataloader, 0):
                real_img_cpu, txt_embedding = data

                real_imgs = real_img_cpu.to(device)
                txt_embedding = txt_embedding.to(device)
  
                # Generate fake images using the generator network
                noise.data.normal_(0, 1)
                inputs = (txt_embedding, noise)
                fake_imgs, mu, logvar = nn.parallel.data_parallel(netG, inputs, device_ids=[device])


                # Update D network
                # todo
                
                # Update G network
                # todo