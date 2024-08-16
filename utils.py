import torch
import torch.nn as nn
import torchvision.utils as vutils

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torch.utils.data as data


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


# def save_img_results_with_desc(data_img, fake_imgs, txt_descriptions, epoch, img_dir):
#     num = 64  # broj slika koje hocemo da sacuvamo
#     images = images[0:num]
#     fake_images = fake_images[0:num]
#     descriptions = txt_descriptions[0:num]
#     print(descriptions)

#     # Create a font object (you may need to specify the path to a .ttf font file)
#     font = ImageFont.load_default()

#     def add_text_to_image(image, text):
#         draw = ImageDraw.Draw(image)
#         text_size = draw.textsize(text, font=font)
#         text_x = (image.width - text_size[0]) / 2
#         text_y = image.height - text_size[1]
#         draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
#         return image
    
#     images_with_text = []
#     for img, desc in zip(fake_imgs, descriptions):
#         img_pil = transforms.ToPILImage()(img.cpu())  # Convert tensor to PIL image
#         img_pil_with_text = add_text_to_image(img_pil, desc)
#         images_with_text.append(transforms.ToTensor()(img_pil_with_text))  # Convert back to tensor

#     images_with_text = torch.stack(images_with_text)

#     if data_img is not None:
#         data_img = data_img[0:num]
#         # Save real images
#         vutils.save_image(data_img, '%s/real_samples.png' % img_dir, normalize=True)

#     # Save fake images with descriptions
#     vutils.save_image(images_with_text, '%s/fake_samples_with_text_epoch_%03d.png' % (img_dir, epoch), normalize=True)

#     if fake_imgs is not None:
#         # Save fake images
#         vutils.save_image(fake_imgs, '%s/fake_samples_epoch_%03d.png' % (img_dir, epoch), normalize=True)


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
    errD_real = criterion(real_features.view(-1), real_labels.view(-1))

    # Pogrešni parovi (realne slike uparene sa pogrešnim uslovima)
    wrong_cond = cond[1:]  # Pomeri uslove za jedan uzorak
    wrong_features = netD(real_imgs[:batch_size-1], wrong_cond)
    errD_wrong = criterion(wrong_features.view(-1), fake_labels[1:].view(-1))
   
    # Lažni parovi
    fake_features = netD(fake, cond)
    errD_fake = criterion(fake_features.view(-1), fake_labels.view(-1))

    
    errD = errD_real + (errD_fake + errD_wrong) * 0.5

    return errD


def generator_loss(netD, fake_imgs, real_labels, conditions):
    criterion = nn.BCELoss()
    cond = conditions.detach()
    
    fake_logits = netD(fake_imgs, cond)
    errD_fake = criterion(fake_logits.view(-1), real_labels.view(-1))
    
    return errD_fake


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD