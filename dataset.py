import torch.utils.data as data
import os.path
import PIL
import pickle
import numpy as np
import random

from PIL import Image

import config as cfg

class TextImageDataset(data.Dataset):
  def __init__(self, data_dir, split='train', embedding_filename = cfg.EMBEDDING_FILENAME, img_size = 64):
    self.img_size = img_size
    self.data = []
    self.data_dir = data_dir
    self.split = split
    split_dir = os.path.join(data_dir, split)

    self.filenames = self.load_filenames(split_dir)
    self.embedding = self.load_embedding(split_dir, embedding_filename)
    self.class_id = self.load_class_id(split_dir, len(self.filenames))

  def load_filenames(self, split_dir):
    filenames_path = os.path.join(split_dir, "filenames.pickle")
    with open(filenames_path, "rb") as f:
      filenames = pickle.load(f)
    return filenames
  
  def load_embedding(self, split_dir, embedding_filename):
    embedding_path = os.path.join(split_dir, embedding_filename)
    with open(embedding_path, "rb") as f:
      embedding = pickle.load(f)
    return embedding

  def load_class_id(self, split_dir):
    with open(split_dir + '/class_info.pickle', 'rb') as f:
      class_id = pickle.load(f, encoding='latin1')   
    return class_id
  


  def get_image(self, path):
    img = Image.open(path)
    img = img.resize((64, 64), PIL.Image.BILINEAR)
    return img
    

# __getitem__ method in Python -> special method that enables instances of a class to use square bracket notation ([]) for accessing element
  def __getitem__(self, index):
    images_path = os.path.join(self.data_dir, "/CUB_200_2011/images")
    img_name = self.filenames[index]
    img_path = os.path.join(images_path, img_name)
    img = self.get_image(img_path)

    embedding = None
    #embedding = self.embedding[index]
    # ??
    return img, embedding
  
