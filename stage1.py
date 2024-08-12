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


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

# up sampling
def upSamplingBlock(in_channels, out_channels):
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
        

        # layers

        self.fc = nn.Sequential(
            nn.Linear(self.z_dim + self.condition_dim, self.gf_dim * 8 * 4 * 4),
            nn.BatchNorm1d(self.gf_dim * 8 * 4 * 4),
            nn.ReLU()
        )
        
        self.upsample1 = upSamplingBlock(self.gf_dim * 8, self.gf_dim * 4)
        self.upsample2 = upSamplingBlock(self.gf_dim * 4, self.gf_dim * 2)
        self.upsample3 = upSamplingBlock(self.gf_dim * 2, self.gf_dim)
        self.upsample4 = upSamplingBlock(self.gf_dim, 3)
        
        self.img = nn.Sequential(
            conv3x3(3, 3),
            nn.Tanh()
        )

    def forward(self, text_embedding, noise):
       
        c, mu, logvar = self.ca_net(text_embedding)

        c = c.to(noise.device)
        # spajanje suma i uslovnog vektora
        input = torch.cat((noise, c), 1)

        x = self.fc(input)
        x = x.view(-1, self.gf_dim * 8, 4, 4)

        # upsampling
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)

        fake_img = self.img(x)

        # vracamo generisane slike i statistike za regularizaciju

        return fake_img, mu, logvar 
    


class Stage1_Discriminator(nn.Module):
    def __init__(self):
        super(Stage1_Discriminator, self).__init__()
        self.df_dim = cfg.GAN_DF_DIM
        self.condition_dim = cfg.GAN_CONDITION_DIM

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = self.conv_block(96, 192)
        self.conv3 = self.conv_block(192, 384)
        self.conv4 = self.conv_block(384, 768)

        self.fc_output_size = 768 * 16 * 16
        self.embed_fc = nn.Linear(self.condition_dim, self.fc_output_size)
        self.embed_bn = nn.BatchNorm1d(self.fc_output_size)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_output_size * 2, 1),
            nn.Sigmoid()
        )


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, image, condition):
        x = self.conv1(image)
        #print(f"After conv1: {x.shape}")

        x = self.conv2(x)
        #print(f"After conv2: {x.shape}")

        x = self.conv3(x)
        #print(f"After conv3: {x.shape}")

        x = self.conv4(x)
        #print(f"After conv4: {x.shape}")

        # Obrada uslovnog vektora
        condition = self.embed_fc(condition)
        #print(f"After embed_fc: {condition.shape}")
        
        condition = self.embed_bn(condition)
        #print(f"After embed_bn: {condition.shape}")
        
        condition = condition.view(-1, 768, 16, 16)
        #print(f"After view: {condition.shape}")

        # Spajanje osobina slike i uslovnog vektora
        #print(f"BEFORE cat X: {x.shape}")
        x = torch.cat((x, condition), 1)
        #print(f"After cat: {x.shape}")
        
        x = self.fc(x)
        #print(f"After fc: {x.shape}")

        return x


class GANTrainer_stage1():
    def __init__(self, output_dir):
        self.model_dir = os.path.join(cfg.DATA_DIR, 'Model_stage1')
        self.image_dir = os.path.join(cfg.DATA_DIR, 'Images_stage1')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.max_epoch = cfg.TRAIN_MAX_EPOCH
        self.batch_size = cfg.TRAIN_BATCH_SIZE
    
    def load_networks(self):
        netG = Stage1_Generator()  
        netG.apply(utils.weight_initialization)

        netD = Stage1_Discriminator()
        netD.apply(utils.weight_initialization)

        return netG, netD


    def train(self, dataloader):
        netG, netD = self.load_networks()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        netG = netG.to(device)
        netD = netD.to(device)
    
        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = torch.FloatTensor(batch_size, nz).to(device)
        fixed_noise = torch.FloatTensor(batch_size, nz).normal_(0, 1).to(device) 
        real_labels = torch.FloatTensor(batch_size).fill_(1).to(device)
        fake_labels = torch.FloatTensor(batch_size).fill_(0).to(device)

        generator_lr = cfg.TRAIN_GENERATOR_LR
        discriminator_lr = cfg.TRAIN_DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN_LR_DECAY_EPOCH

        optimizerG = Adam(netG.parameters(), lr=generator_lr)
        optimizerD = Adam(netD.parameters(), lr=discriminator_lr)
        
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
  
                # Generisanje laznih slika pomocu generatora G
                noise.data.normal_(0, 1)
                inputs = (txt_embedding, noise)
                fake_imgs, mu, logvar = netG(*inputs)


                ############################
                # (1) Azuriraj D mrezu (diskriminator)
                ###########################
                netD.zero_grad()
                errD = utils.discriminator_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, mu)
                errD.backward()
                optimizerD.step()


                ############################
                # (2) Azuriraj G mrezu (generator)
                ###########################
                netG.zero_grad()
                errG = utils.generator_loss(netD, fake_imgs, real_labels, mu)
                kl_loss = utils.KL_loss(mu, logvar)
                errG_total = errG + kl_loss * cfg.TRAIN_COEFF_KL
                errG_total.backward()
                optimizerG.step()

                if (epoch * len(dataloader) + i) % 100 == 0:
                    inputs = (txt_embedding, fixed_noise)
                    fake_imgs, _, _ = netG(*inputs)  # Pozivamo generator i dobijamo samo fake slike
                    utils.save_img_results(real_img_cpu, fake_imgs, epoch, self.image_dir)
        
                    
            utils.save_model(netG, netD, self.max_epoch, self.model_dir)
