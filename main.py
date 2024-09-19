import config as cfg
from dataset import TextImageDataset
from stage1 import GANTrainer_stage1
from stage2 import GANTrainer_stage2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

class Args:
    train = 1
    stage = 1


if Args.stage == 1:
    cfg.IMG_SIZE = 64
    cfg.NET_G = ''
    cfg.NET_D = ''
    #cfg.TRAIN_MAX_EPOCH = 300
else:
    cfg.IMG_SIZE = 256
    cfg.GAN_GF_DIM = 256

    #cfg.TRAIN_MAX_EPOCH = 300
    cfg.STAGE1_G = "data_reduced1/birds/Model_stage1/netG_epoch_300.pth"
    cfg.STAGE1_D = "data_reduced1/birds/Model_stage1/netD_epoch_300.pth"

# za 40 vrsti:
# cfg.TRAIN_DISCRIMINATOR_LR = 0.00005
# cfg.TRAIN_GENERATOR_LR = 0.0002

cfg.TRAIN_DISCRIMINATOR_LR = 0.0002
cfg.TRAIN_GENERATOR_LR = 0.0009 # RADI FINO ZA GEN LR = 0.0005


cfg.STAGE = Args.stage

if Args.train == 1:
    cfg.TRAIN = True
else:
    cfg.NET_G = "../data_reduced_20/birds/Model_stage2/netG_epoch_200.pth"
    cfg.NET_D = "../data_reduced_20/birds/Model_stage2/netD_epoch_200.pth"
    cfg.TRAIN = False


if torch.cuda.is_available():
  cfg.DEVICE = 'cuda'
else:
  cfg.DEVICE = 'cpu'

output_dir = '../output/birds'


def train():
  image_transform = transforms.Compose([
      transforms.RandomResizedCrop(cfg.IMG_SIZE),  # Randomly crop and resize the image
      transforms.RandomHorizontalFlip(),         # Randomly flip the image horizontally
      transforms.ColorJitter(),                  # Randomly adjust brightness, contrast, saturation, and hue
      transforms.ToTensor(),                     # Convert the image to a PyTorch tensor
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
  ])

  image_transform2 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),  # Minimalna šansa za horizontalno ogledanje slike
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),  # Vrlo mala promena osvetljenosti, kontrasta, saturacije i nijanse
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalizacija slike
  ])

  image_transform3 = transforms.Compose([
            transforms.RandomCrop(cfg.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

  dataset = TextImageDataset(cfg.DATA_DIR, 'train', input_transform=image_transform)
  dataset_size = len(dataset)
  #print(f"Veličina skupa podataka: {dataset_size}")

  dataloader = DataLoader(dataset, batch_size=cfg.TRAIN_BATCH_SIZE, drop_last=True, shuffle=False, num_workers=4)
  batch_size = dataloader.batch_size
  #print(f"Veličina batch-a: {batch_size}")

  num_batches_per_epoch = len(dataloader)
  #print(f"Broj batch-eva po epohi: {num_batches_per_epoch}")

  if Args.stage == 1:
    trainer = GANTrainer_stage1(output_dir)
  else:
    trainer = GANTrainer_stage2(output_dir)

  trainer.train(dataloader)

def test():
    image_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = TextImageDataset(cfg.DATA_DIR, 'test', embedding_filename=cfg.EMBEDDING_FILENAME, transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=cfg.TRAIN_BATCH_SIZE, drop_last=True, shuffle=True, num_workers=4)
    N = len(dataloader)

    if Args.stage == 1:
        trainer = GANTrainer_stage1(output_dir)
    else:
        trainer = GANTrainer_stage2(output_dir)

    trainer.test(dataloader)

    dataset_generated = TextImageDataset(cfg.DATA_DIR, 'test', imsize=cfg.IMSIZE, transform=image_transform,
                                            embedding_filename=cfg.EMBEDDING_FILENAME)
    dataloader_generated = torch.utils.data.DataLoader(dataset_generated, batch_size=cfg.TRAIN_BATCH_SIZE, drop_last=True, shuffle=True, num_workers=4)


def main():
    if Args.train == 0:
        cfg.TRAIN = False  # True by default (train)
    else:
        cfg.TRAIN = True

    if Args.stage == 1:
        cfg.IMG_SIZE = 64
        cfg.STAGE = 1
    else :
        cfg.IMG_SIZE = 256
        cfg.STAGE = 2

    output_dir = '../output/birds'


    if Args.train:
        train()
    else:
        test()

main()