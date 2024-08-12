import torch
import torch.nn as nn
import os
from torch.optim import Adam

import config as cfg
import utils
import conditioning_augmentation as ca
from stage1 import Stage1_Generator
import torch.nn.functional as F



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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # dodavanje inputa (residual connection)
        out = self.relu(out)
        return out


## proveeeri
def concat_along_dims(c, x):
    """Joins the conditioned text with the encoded image along the dimensions."""
    c = c.unsqueeze(2).unsqueeze(3)  # Adding two dimensions
    c = c.expand(-1, -1, x.size(2), x.size(3))  # Expand along height and width
    return torch.cat([c, x], dim=1)  # Concatenate along the channel dimension



class Stage2_Generator(nn.Module):
    def __init__(self, Stage1_Generator):
        super(Stage2_Generator, self).__init__()
        self.gf_dim = cfg.GAN_GF_DIM
        self.condition_dim = cfg.GAN_CONDITION_DIM
        self.z_dim = cfg.Z_DIM
        self.Stage1_G = Stage1_Generator

        # Stage 1 model parameters are frozen
        for param in self.Stage1_G.parameters():
            param.requires_grad = False

        self.ca_net = ca.CANet()

        # Encoding image from Stage1
        self.encoder = nn.Sequential(
            conv3x3(3, self.gf_dim),
            nn.ReLU(True),
            nn.Conv2d(self.gf_dim, self.gf_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.gf_dim * 2),
            nn.ReLU(True),
            nn.Conv2d(self.gf_dim * 2, self.gf_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.gf_dim * 4),
            nn.ReLU(True))
        
        # Combining encoded image with text embedding
        self.hr_joint = nn.Sequential(
            conv3x3(self.condition_dim + self.gf_dim * 4, self.gf_dim * 4),
            nn.BatchNorm2d(self.gf_dim * 4),
            nn.ReLU(True)
        )

        self.residual = nn.Sequential(
            ResidualBlock(self.gf_dim * 4),
            ResidualBlock(self.gf_dim * 4)
        )
        
        self.upsample1 = upSamplingBlock(self.gf_dim * 4, self.gf_dim * 2)
        self.upsample2 = upSamplingBlock(self.gf_dim * 2, self.gf_dim)
        self.upsample3 = upSamplingBlock(self.gf_dim, self.gf_dim // 2)
        self.upsample4 = upSamplingBlock(self.gf_dim // 2, self.gf_dim // 4)

        self.img = nn.Sequential(
            conv3x3(self.gf_dim // 4, 3),
            nn.Tanh())
        
    def forward(self, text_embedding, noise):
        stage1_img, _, _ = self.Stage1_G(text_embedding, noise)
        stage1_img = stage1_img.detach()

        # encode stage1 image 
        encoded_img = self.encoder(stage1_img)

        c_code, mu, logvar = self.ca_net(text_embedding)
        c_code = c_code.view(-1, self.condition_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 16, 16)
        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        fake_img = self.img(h_code)
        return fake_img, mu, logvar


class Stage2_Discriminator(nn.Module):
    def __init__(self):
        super(Stage2_Discriminator, self).__init__()
        self.df_dim = cfg.GAN_DF_DIM
        self.condition_dim = cfg.GAN_CONDITION_DIM

        # Konvolucioni slojevi za obradu slike
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
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Konvolucioni slojevi za obradu uslovnog vektora
        self.encode_condition = nn.Sequential(
            nn.Linear(self.condition_dim, self.df_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(self.df_dim * 8 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Završni sloj za kombinovanje uslovnog vektora i slike
        self.fc = nn.Sequential(
            nn.Conv2d(self.df_dim * 8 + self.df_dim * 8, self.df_dim * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.df_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Dodavanje globalnog average pooling sloja


    def forward(self, image, condition):
        img_embedding = self.encode_img(image)
        batch_size, _, height, width = img_embedding.size()

        cond_embedding = self.encode_condition(condition)
        cond_embedding = cond_embedding.view(batch_size, self.df_dim * 8, 4, 4)
        cond_embedding = F.interpolate(cond_embedding, size=(height, width), mode='bilinear', align_corners=False)

        joint_input = torch.cat((img_embedding, cond_embedding), 1)
        output = self.fc(joint_input)

        output = self.pool(output)  # Global average pooling
        output = output.view(batch_size, -1)  # Flattening da bi dimenzije bile [batch_size, 1]

        return output



class GANTrainer_stage2():
    def __init__(self, output_dir):
        self.model_dir = os.path.join(cfg.DATA_DIR, 'Model_stage2')
        self.image_dir = os.path.join(cfg.DATA_DIR, 'Images_stage2')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.max_epoch = cfg.TRAIN_MAX_EPOCH
        self.batch_size = cfg.TRAIN_BATCH_SIZE
    
    def load_networks(self):
        Stage1_G = Stage1_Generator()
        netG = Stage2_Generator(Stage1_G)  
        netG.apply(utils.weight_initialization)

        netD = Stage2_Discriminator()
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

                #print(f"Epoch [{epoch}/{self.max_epoch}], Batch [{i}/{len(dataloader)}], Discriminator Loss: {errD.item()}")

                ############################
                # (2) Azuriraj G mrezu (generator)
                ###########################
                netG.zero_grad()
                errG = utils.generator_loss(netD, fake_imgs, real_labels, mu)
                kl_loss = utils.KL_loss(mu, logvar)
                errG_total = errG + kl_loss * cfg.TRAIN_COEFF_KL
                errG_total.backward(retain_graph=True)
                optimizerG.step()


                if (epoch * len(dataloader) + i) % 100 == 0:
                    inputs = (txt_embedding, fixed_noise)
                    fake_imgs, _, _ = netG(*inputs)  # Pozivamo generator i dobijamo samo fake slike
                    utils.save_img_results(real_img_cpu, fake_imgs, epoch, self.image_dir)

                torch.cuda.empty_cache()

                    
            utils.save_model(netG, netD, self.max_epoch, self.model_dir)
