import torch
import torch.nn as nn
import os
from torch.optim import Adam

import config as cfg
import utils
import conditioning_augmentation as ca
from stage1 import Stage1_Generator
import torch.nn.functional as F

generator_losses = []
discriminator_losses = []

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
        self.block = nn.Sequential(
            conv3x3(in_channels, in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            conv3x3(in_channels, in_channels),
            nn.BatchNorm2d(in_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual # dodavanje inputa (residual connection)
        out = self.relu(out)
        return out


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
        _, stage1_img, _, _ = self.Stage1_G(text_embedding, noise)
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
        #print(f"fake image stage 2 shape{fake_img.shape}")
        return stage1_img, fake_img, mu, logvar


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

            nn.Conv2d(self.df_dim * 8, self.df_dim * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.df_dim * 16, self.df_dim * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 32),
            nn.LeakyReLU(0.2, inplace=True),

            conv3x3(self.df_dim * 32, self.df_dim * 16),
            nn.BatchNorm2d(self.df_dim * 16),
            nn.LeakyReLU(0.2, inplace=True), 

            conv3x3(self.df_dim * 16, self.df_dim * 8),
            nn.BatchNorm2d(self.df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True) 
        )
        
        # slojevi za obradu uslovnog vektora
        self.embed_fc = nn.Linear(self.condition_dim, self.df_dim * 8 * 4 * 4)
        self.embed_bn = nn.BatchNorm2d(self.df_dim * 8)
      
        # Završni sloj za kombinovanje uslovnog vektora i slike
        self.fc = nn.Sequential(
            nn.Conv2d(self.df_dim * 8 + self.df_dim * 8, self.df_dim * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.df_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)


    def forward(self, image, condition):
        # Obrada slike
        img_embedding = self.encode_img(image)

        # Obrada uslovnog vektora
        condition = self.embed_fc(condition)
        condition = condition.view(-1, self.df_dim * 8, 4, 4)
        condition = self.embed_bn(condition)

        # Spajanje slike i uslovnog vektora
        h_c_code = torch.cat((img_embedding, condition), 1)

        # Finalna procena
        output = self.fc(h_c_code)
        return output.view(-1)



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

        if cfg.NET_G != '':
            state_dict = torch.load(cfg.NET_G, map_location=torch.device('cpu'))
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        elif cfg.STAGE1_G != '':
            state_dict = torch.load(cfg.STAGE1_G, map_location=torch.device('cpu'))
            netG.Stage1_G.load_state_dict(state_dict)
            print('Load from: ', cfg.STAGE1_G)
        else:
            print("Please give the Stage1_G path")
            return

        netD = Stage2_Discriminator()
        netD.apply(utils.weight_initialization)
        if cfg.NET_D != '':
            state_dict = torch.load(cfg.NET_D, map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)

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

        lr_decay_step = cfg.TRAIN_LR_DECAY_EPOCH # posle koliko epoha se koeficijent ucenja smanjuje
        generator_lr = cfg.TRAIN_GENERATOR_LR
        discriminator_lr = cfg.TRAIN_DISCRIMINATOR_LR

        optimizerG = Adam(netG.parameters(), lr=cfg.TRAIN_GENERATOR_LR, betas=(0.5, 0.999))
        optimizerD = Adam(netD.parameters(), lr=cfg.TRAIN_DISCRIMINATOR_LR, betas=(0.5, 0.999) )
        
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
                real_img_cpu, txt_embedding, txt_descriptions = data
                
                real_imgs = real_img_cpu.to(device)
                txt_embedding = txt_embedding.to(device)
  
                # Generisanje laznih slika pomocu generatora G
                #noise = torch.randn(batch_size, nz, device=device)

                noise.data.normal_(0, 1)
                inputs = (txt_embedding, noise)
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

                # za svaku epohu, prolazimo kroz dataloader i cuvamo samo prvu instancu podataka za i=0, tj mi tehnicki pratimo (cuvamo) prvi batch i kako tece njegovo treniranje
                if i == 0:
                    with torch.no_grad(): 
                        inputs = (txt_embedding, fixed_noise)
                        fake_source_img, fake_imgs, _, _ = netG(*inputs)
                        utils.save_img_results(real_img_cpu, fake_imgs.detach(), epoch, self.image_dir)
                        if fake_source_img is not None:
                            utils.save_img_results(None, fake_source_img, epoch, self.image_dir)
            
            avg_G_loss = total_G_loss / batch_count
            avg_D_loss = total_D_loss / batch_count
            
            generator_losses.append(errG_total.item())
            discriminator_losses.append(errD.item())

            if epoch == cfg.TRAIN_MAX_EPOCH - 1:
                print(f"Descriptions for epoch {epoch+1}:")
                print(descriptions)
                print("\n")

            utils.save_model(netG, netD, self.max_epoch, self.model_dir)

        utils.plot_losses(generator_losses, discriminator_losses)

    def test(self, dataloader, stage=1):
        netG, _ = self.load_networks()

        # path to save generated samples
        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')]
        if not os.path.isdir(save_dir):
            mkdir_p(save_dir)

        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))

        count = 0
        for i, data in enumerate(data_loader, 0):
            _, txt_embedding = data
            txt_embedding = Variable(txt_embedding)

            if cfg.CUDA:
                txt_embedding = txt_embedding.cuda()

            noise.data.normal_(0, 1)
            inputs = (txt_embedding, noise)

            _, fake_imgs, mu, logvar = netG(*inputs)

            utils.save_test_results(fake_imgs, count, save_dir)

            # for i in range(batch_size):
            #     save_name = '%s/%d.png' % (save_dir, count + i)
            #     im = fake_imgs[i].data.cpu().numpy()
            #     im = (im + 1.0) * 127.5
            #     im = im.astype(np.uint8)
            #     im = np.transpose(im, (1, 2, 0))
            #     im = Image.fromarray(im)
            #     im.save(save_name)
            count += batch_size