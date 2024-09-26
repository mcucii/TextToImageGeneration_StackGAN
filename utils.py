import torch
import torch.nn as nn
import torchvision.utils as vutils

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torch.utils.data as data
from urllib.request import urlopen
import torch.autograd as autograd

import matplotlib.pyplot as plt
import os


def weight_initialization(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

# def weight_initialization_xavier(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
#     elif classname.find('BatchNorm') != -1:
#         nn.init.ones_(m.weight.data)
#         nn.init.zeros_(m.bias.data)
#     elif classname.find('Linear') != -1:
#         nn.init.xavier_normal_(m.weight.data)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias.data)

# def weight_initialization_he(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
#     elif classname.find('BatchNorm') != -1:
#         nn.init.ones_(m.weight.data)
#         nn.init.zeros_(m.bias.data)
#     elif classname.find('Linear') != -1:
#         nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.zeros_(m.bias.data)


            

def save_model(netG, netD, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_epoch_%d.pth' % (model_dir, epoch))
    print('Save G/D models')


def save_test_results(fake_imgs, count, img_dir):
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    vutils.save_image(fake_imgs, '%s/fake_samples_test_%03d.png' % (img_dir, count), normalize=True)


def save_img_results(data_img, fake_imgs, epoch, img_dir):
    if data_img is not None:
        # vutils.save_image kombinuje slike iz batch-a u jednu sliku
        vutils.save_image(data_img, '%s/real_samples.png' % img_dir, normalize=True)
        vutils.save_image(fake_imgs, '%s/fake_samples_epoch_%03d.png' % (img_dir, epoch), normalize=True)
    else:
        vutils.save_image(fake_imgs, '%s/fake_samples_epoch_%03d_stage1.png' % (img_dir, epoch), normalize=True)


def discriminator_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, conditions):
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)

    cond = conditions.detach()
    fake = fake_imgs.detach()

    # Real parovi
    real_features = netD(real_imgs, cond)
    # print(f'REAL FEATURES BY DISC: {real_features}\n')
    # print(f'REAL LABELS: {real_labels}\n')
    errD_real = criterion(real_features.view(-1), real_labels.view(-1))

    # Pogrešni parovi (realne slike uparene sa pogrešnim random uslovima)
    wrong_cond = cond[torch.randperm(batch_size)]  # Slučajna permutacija
    wrong_features = netD(real_imgs, wrong_cond)
    #print(f'WRONG FEATURES BY DISC - real_imgs fake_cond: {wrong_features}\n')
    errD_wrong = criterion(wrong_features.view(-1), fake_labels.view(-1))
   
    # Lažni parovi
    fake_features = netD(fake, cond)
    errD_fake = criterion(fake_features.view(-1), fake_labels.view(-1))
    #print(f'FAKE FEATURES BY DISC - fake_imgs real_cond: {fake_features}\n')

    errD = errD_real + (errD_fake + errD_wrong)*0.5

    return errD


def generator_loss(netD, fake_imgs, real_labels, conditions, logvar):
    criterion = nn.BCELoss()
    # detach() se ne koristi jer gradijenti trebaju biti uključeni kako bi se pravilno obučio generator
    cond = conditions.detach()
    fake = fake_imgs
    
    fake_features = netD(fake, cond)
    errD_fake = criterion(fake_features.view(-1), real_labels.view(-1))
    #errD_fake = -torch.mean(torch.log(fake_features + 1e-8))  # Izbegavanje log(0) dodavanjem malog epsilon-a

    kl_loss = KL_loss(conditions, logvar)

    errG_total = errD_fake + kl_loss*2

    return errG_total


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def plot_losses(generator_losses, discriminator_losses, max_epoch):
    plt.figure(figsize=(10, 5))
    
    # Plotovanje gubitaka generatora
    plt.subplot(1, 2, 1)
    plt.plot(generator_losses, label='Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Generator Loss per Epoch')
    plt.legend()
    
    # Plotovanje gubitaka diskriminatora
    plt.subplot(1, 2, 2)
    plt.plot(discriminator_losses, label='Discriminator Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss per Epoch')
    plt.legend()
    
    plt.xlim(0, max_epoch)  # Set x-axis limit based on max_epoch
    plt.tight_layout()
    plt.show()

