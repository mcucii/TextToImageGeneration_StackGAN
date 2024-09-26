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

import matplotlib.pyplot as plt


generator_losses = []
discriminator_losses = []


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

def upSamplingBlock(in_channels, out_channels):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
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
            nn.Linear(self.z_dim + self.condition_dim, self.gf_dim * 4 * 4, bias=False),
            nn.BatchNorm1d(self.gf_dim * 4 * 4),
            nn.ReLU(True)
        )
        
        self.upsample1 = upSamplingBlock(self.gf_dim, self.gf_dim // 2)
        self.upsample2 = upSamplingBlock(self.gf_dim // 2, self.gf_dim // 4)
        self.upsample3 = upSamplingBlock(self.gf_dim // 4, self.gf_dim // 8)
        self.upsample4 = upSamplingBlock(self.gf_dim // 8, self.gf_dim // 16)

        self.img = nn.Sequential(
            conv3x3(self.gf_dim // 16 , 3),
            nn.Tanh()
        )

    def forward(self, text_embedding, noise):
        c, mu, logvar = self.ca_net(text_embedding)

        c.to(cfg.DEVICE)

        # spajanje suma i uslovnog vektora
        input = torch.cat((noise, c), 1)
        x = self.fc(input)
        
        x = x.view(-1, self.gf_dim, 4, 4)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)

        fake_img = self.img(x)

        # vracamo generisane slike i statistike za regularizaciju
        return None, fake_img, mu, logvar 
    

class Stage1_Discriminator(nn.Module):
    def __init__(self):
        super(Stage1_Discriminator, self).__init__()
        self.df_dim = cfg.GAN_DF_DIM
        self.condition_dim = cfg.GAN_CONDITION_DIM

        self.encode_img = nn.Sequential(
            nn.Conv2d(3, self.df_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.df_dim, self.df_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.df_dim * 2, self.df_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.df_dim * 4, self.df_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Sloj za obradu uslovnog vektora
        self.embed_fc = nn.Linear(self.condition_dim, self.df_dim * 8 * 4 * 4)
        self.embed_bn = nn.BatchNorm1d(self.df_dim * 8 * 4 * 4)

        # Sloj za finalnu procenu
        self.outlogits = nn.Sequential(
            nn.Conv2d(self.df_dim * 8 + self.df_dim * 8, self.df_dim * 8, kernel_size=1, padding=1),
            nn.BatchNorm2d(self.df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.df_dim * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid()
        )

    def forward(self, image, condition):
        # Obrada slike
        h_code = self.encode_img(image)

        # Obrada uslovnog vektora
        condition = self.embed_fc(condition)
        condition = self.embed_bn(condition)
        condition = condition.view(-1, self.df_dim * 8, 4, 4)

        # Spajanje slike i uslovnog vektora
        h_c_code = torch.cat((h_code, condition), 1)

        # Finalna procena
        output = self.outlogits(h_c_code)
        return output.view(-1)


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

        netG.to(cfg.DEVICE)
        netD.to(cfg.DEVICE)
        
        return netG, netD


    def train(self, dataloader):
        netG, netD = self.load_networks()

        device = cfg.DEVICE
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

        optimizerG = torch.optim.Adam(netG.parameters(), lr=generator_lr, betas=(0.5, 0.999))
        optimizerD = torch.optim.Adam(netD.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))
        #optimizerG = torch.optim.Adam(netG.parameters(), lr=generator_lr, betas=(0.5, 0.999))
        #optimizerD = torch.optim.SGD(netD.parameters(), lr=discriminator_lr, momentum=0.9)

        lambda_gp = 10

        
        for epoch in range(self.max_epoch):
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr

                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr     

            descriptions = []       

            total_G_loss = 0.0
            total_D_loss = 0.0
            batch_count = 0

            for i, data in enumerate(dataloader, 0):
                real_img_cpu, img_embeddings, txt_descriptions = data
                #real_img_cpu, img_embeddings = data

                real_imgs = real_img_cpu.to(device)
                img_embeddings = img_embeddings.to(device)
  
                # Generisanje laznih slika pomocu generatora G
                noise.normal_(0, 1)
                inputs = (img_embeddings, noise)
                _, fake_imgs, mu, logvar = netG(*inputs)

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
                errG_total = utils.generator_loss(netD, fake_imgs, real_labels, mu, logvar)
                errG_total.backward()
                optimizerG.step()

                total_G_loss += errG_total.item()
                total_D_loss += errD.item()

                batch_count += 1
                descriptions = txt_descriptions

                print(f'Epoch [{epoch}/{self.max_epoch}], Step [{i}/{len(dataloader)}], '
                       f'Generator Loss: {errG_total.item()}, Discriminator Loss: {errD.item()}')

                if i == 0:
                    with torch.no_grad(): 
                        inputs = (img_embeddings, fixed_noise)
                        _, fake_imgs, _, _ = netG(*inputs)
                        utils.save_img_results(real_img_cpu, fake_imgs.detach(), epoch, self.image_dir)
                      
            avg_G_loss = total_G_loss / batch_count
            avg_D_loss = total_D_loss / batch_count
            
            generator_losses.append(avg_G_loss)
            discriminator_losses.append(avg_D_loss)

            if epoch == cfg.TRAIN_MAX_EPOCH - 1:
                print(f"Descriptions for epoch {epoch+1}:")
                print(descriptions)
                print("\n")
                    
            utils.save_model(netG, netD, self.max_epoch, self.model_dir)

        utils.plot_losses(generator_losses, discriminator_losses, self.max_epoch)

    def test(self, dataloader, stage=1):
        netG, _ = self.load_networks()

        # path to save generated samples
        save_dir = os.path.join(cfg.DATA_DIR, 'test')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        device = cfg.DEVICE

        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = torch.FloatTensor(batch_size, nz).to(device)


        count = 0
        for i, data in enumerate(dataloader, 0):
            _, img_embeddings, txt_descriptions = data
                #real_img_cpu, img_embeddings = data

            img_embeddings = img_embeddings.to(device)

            noise.normal_(0, 1)
            inputs = (img_embeddings, noise)

            _, fake_imgs, mu, logvar = netG(*inputs)

            utils.save_test_results(fake_imgs, count, save_dir)
            print(f"Descriptions batch {i}: {txt_descriptions} \n")

            # for i in range(batch_size):
            #     save_name = '%s/%d.png' % (save_dir, count + i)
            #     im = fake_imgs[i].data.cpu().numpy()
            #     im = (im + 1.0) * 127.5
            #     im = im.astype(np.uint8)
            #     im = np.transpose(im, (1, 2, 0))
            #     im = Image.fromarray(im)
            #     im.save(save_name)
            count += batch_size




