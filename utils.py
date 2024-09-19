import torch
import torch.nn as nn
import torchvision.utils as vutils

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torch.utils.data as data
from urllib.request import urlopen
import torch.autograd as autograd

import matplotlib.pyplot as plt





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
        '%s/netD_epoch_%d.pth' % (model_dir, epoch))
    print('Save G/D models')


def save_img_results_with_desc(data_img, fake_images, txt_descriptions, epoch, img_dir):
    num = 64  # broj slika koje hocemo da sacuvamo
    data_img = data_img[0:num]
    fake_images = fake_images[0:num]
    descriptions = txt_descriptions[0:num]

    font_size = 20  # Povećaj veličinu fonta za bolju vidljivost
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Koristi jasan font
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()  # Ako ne uspe, koristi podrazumevani font

    def add_text_to_image(image, text):
        image = image.convert("RGB")  # Pretvori u RGB format da izbegneš neprirodne boje
        draw = ImageDraw.Draw(image)

        # Dobij veličinu teksta za centriranje
        text_size = draw.textsize(text, font=font)

        # Kreiraj novu sliku sa dodatnim prostorom za tekst
        new_image = Image.new("RGB", (image.width, image.height + text_size[1] + 10), (0, 0, 0))
        new_image.paste(image, (0, 0))

        # Pozicioniraj tekst na dnu slike
        text_x = (new_image.width - text_size[0]) / 2
        text_y = image.height
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))  # Bela boja teksta za vidljivost
        return new_image

    images_with_text = []
    for img, desc in zip(fake_images, descriptions):
        img_pil = transforms.ToPILImage()(img.cpu())  # Konvertuj tenzor u PIL sliku
        img_pil_with_text = add_text_to_image(img_pil, desc)
        images_with_text.append(transforms.ToTensor()(img_pil_with_text))  # Konvertuj nazad u tenzor

    images_with_text = torch.stack(images_with_text)

    # Sačuvaj slike bez normalizacije
    vutils.save_image(images_with_text, '%s/fake_samples_with_text_epoch_%03d.png' % (img_dir, epoch))


def save_test_results(fake_imgs, count, img_dir):
    # Kreiraj direktorijum ako ne postoji
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    # Sačuvaj generisane slike
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
    #errD = errD_real

    return errD


def generator_loss(netD, fake_imgs, real_labels, conditions, logvar):
    criterion = nn.BCELoss()
    # detach() se ne koristi jer gradijenti trebaju biti uključeni kako bi se pravilno obučio generator
    cond = conditions.detach()
    fake = fake_imgs
    
    fake_features = netD(fake, cond)
    errD_fake = criterion(fake_features.view(-1), real_labels.view(-1))
    #print(f'FAKE DISC OUTPUT IN GEN - fake imgs, real cond: {fake_features}\n\n')
    
    kl_loss = KL_loss2(conditions, logvar)
    errG_total = errD_fake + kl_loss*2

    #return errD_fake
    return errG_total


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def KL_loss2(mu, logvar):
    # Pretvaranje logvar u varijansu (sigma^2)
    sigma_sq = torch.exp(logvar)
    
    # Izračunavanje KL divergence
    KLD_element = -0.5 * (1 + logvar - mu.pow(2) - sigma_sq)
    
    # Uzmi srednju vrednost preko svih dimenzija
    KLD = torch.mean(KLD_element)
    
    return KLD


def plot_losses(generator_losses, discriminator_losses):
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
    
    plt.tight_layout()
    plt.show()
