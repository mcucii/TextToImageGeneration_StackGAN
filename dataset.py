import torch.utils.data as data
import os.path
import PIL
import pickle
import numpy as np
import random


import config as cfg

class TextImageDataset(data.Dataset):
  def __init__(self, data_dir, split='train', embedding_filename = cfg.EMBEDDING_FILENAME, img_size = 64):
    self.img_size = img_size
    self.data = []
    self.data_dir = data_dir
    split_dir = os.path.join(data_dir, split)

    self.filenames = self.load_filenames(split_dir)
    self.embeddings = self.load_embedding(split_dir, embedding_filename)
    self.class_id = self.load_class_id(split_dir, len(self.filenames))

  def load_filenames(split_dir):
    pass

  def load_embedding(split_dir, embedding_filename):
    pass

  def load_class_id(split_dir):
    pass
