import torch
import torch.nn as nn
import torchvision.utils as vutils

def weight_initialization(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def save_model(netG, netD, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_epoch_last.pth' % (model_dir))
    print('Save G/D models')


def save_img_results(data_img, fake_imgs, epoch, img_dir):
    num = 64 # broj slika koje hocemo da sacuvamo
    fake_imgs = fake_imgs[0:num]
    if data_img is not None:
        data_img = data_img[0:num]
        # vutils.save_image kombinuje slike iz batch-a u jednu sliku
        vutils.save_image(data_img, '%s/real_samples.png' % img_dir, normalize=True)
        vutils.save_image(fake_imgs, '%s/fake_samples_epoch_%03d.png' % (img_dir, epoch), normalize=True)
    else:
        vutils.save_image(fake_imgs, '%s/fake_samples_epoch_%03d.png' % (img_dir, epoch), normalize=True)




def discriminator_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, conditions):
    criterion = nn.BCELoss()
    cond = conditions.detach()

    real_logits = netD(real_imgs, cond)
    real_labels = real_labels.unsqueeze(1).float()
    errD_real = criterion(real_logits, real_labels)

    fake_logits = netD(fake_imgs, cond)
    fake_labels = fake_labels.unsqueeze(1).float()
    errD_fake = criterion(fake_logits, fake_labels)

    errD = errD_real + errD_fake
    return errD


def generator_loss(netD, fake_imgs, real_labels, conditions):
    criterion = nn.BCELoss()
    cond = conditions.detach()
    
    fake_logits = netD(fake_imgs, cond)
    real_labels = real_labels.unsqueeze(1)
    errD_fake = criterion(fake_logits, real_labels)
    
    return errD_fake


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD