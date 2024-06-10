import torch.nn as nn

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


def discriminatior_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, mu, gpus):
    criterion = nn.BCELoss()
    device = real_imgs.device
    real_labels = real_labels.to(device)
    fake_labels = fake_labels.to(device)

    real_outputs = nn.parallel.data_parallel(netD, real_imgs, device_ids=gpus)
    errD_real = criterion(real_outputs, real_labels)

    fake_outputs = nn.parallel.data_parallel(netD, fake_imgs, device_ids=gpus)
    errD_fake = criterion(fake_outputs, fake_labels)
               
    return errD_real, errD_fake, errD_real.data, errD_fake.data



def generator_loss(netD, fake_imgs, real_labels, conditions, gpus):
    criterion = nn.BCELoss()
    device = fake_imgs.device
    real_labels = real_labels.to(device)

    outputs = nn.parallel.data_parallel(netD, fake_imgs, device_ids=gpus)

    # The generator loss - racuna se na osnovu toga koliko dobro diskriminator moze da oceni da su fejk slike stvarno fejk! 
    errD_fake = criterion(outputs, real_labels)

    return errD_fake
