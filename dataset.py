import torch.utils.data as data
import os.path
import PIL
import pickle
import numpy as np
import random
import torchvision.transforms as transforms


from PIL import Image

import config as cfg

class TextImageDataset(data.Dataset):
  def __init__(self, data_dir, split='train', embedding_filename = cfg.EMBEDDING_FILENAME, img_size = 64, input_transform=None,
                 target_transform=None):
    self.img_size = img_size
    self.data = []
    self.data_dir = data_dir
    self.split = split
    split_dir = os.path.join(data_dir, split)
    self.input_transform = input_transform
    self.target_transform = target_transform
    self.description_folder = os.path.join(data_dir, 'text_c10')

    self.filenames = self.load_filenames(split_dir)
    self.embeddings = self.load_embeddings(split_dir, embedding_filename)
    self.descriptions = self.load_descriptions()
    self.class_id = self.load_class_id(split_dir, len(self.filenames))

    if self.input_transform is None:
            self.input_transform = transforms.ToTensor()

  def load_filenames(self, split_dir):
    filenames_path = os.path.join(split_dir, "filenames.pickle")
    with open(filenames_path, 'rb') as f:
      filenames = pickle.load(f, encoding='latin1')
    return filenames
  
  def load_embeddings(self, split_dir, embedding_filename):
    embedding_path = os.path.join(split_dir, embedding_filename)
    with open(embedding_path, 'rb') as f:
      embeddings = pickle.load(f, encoding='latin1')
    return embeddings
  
  def load_descriptions(self):
    descriptions = {}
    for bird_type_folder in os.listdir(self.description_folder):
      bird_type_path = os.path.join(self.description_folder, bird_type_folder)
      if os.path.isdir(bird_type_path):
        for filename in os.listdir(bird_type_path):
          if filename.endswith('.txt'):
            file_path = os.path.join(bird_type_path, filename)
            with open(file_path, 'r') as file:
              description = file.read().strip()
              image_name = filename.replace('.txt', '.jpg')  
              descriptions[image_name] = description
    return descriptions
  

  def load_class_id(self, split_dir, total_num):
    with open(split_dir + '/class_info.pickle', 'rb') as f:
      class_id = pickle.load(f, encoding='latin1')   
    return class_id
  
  def get_image(self, path):
    img = Image.open(path).convert("RGB");
    img = img.resize((cfg.IMG_SIZE, cfg.IMG_SIZE), PIL.Image.BILINEAR)
    if self.input_transform is not None:
      img = self.input_transform(img)
    return img
    

# __getitem__ method in Python -> special method that enables instances of a class to use square bracket notation ([]) for accessing element
  def __getitem__(self, index):
    images_path = os.path.join(self.data_dir, "CUB_200_2011/images")
    img_name = self.filenames[index] + ".jpg"
    img_path = os.path.join(images_path, img_name)
    img = self.get_image(img_path)

    img_embeddings = self.embeddings[index][:][:]

    if img_name in self.descriptions:
      img_txt_description = self.descriptions[img_name]  
    else:
      img_txt_description = "Opis nije pronađen" 

    #print(f'img_embeddings_shape : {img_embeddings.shape[0]}')

    rnd_idx = random.randint(0, img_embeddings.shape[0] - 1)
    rnd_img_embedding = img_embeddings[rnd_idx, :]
    #rnd_txt_description = img_txt_description[rnd_idx, :]

    return img, rnd_img_embedding
    #return img, rnd_img_embedding, rnd_txt_description
  
  def __len__(self):
        return len(self.filenames)
