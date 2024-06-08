import torch
import torch.nn as nn

import config as cfg

# CANet transformiše tekstualne embedding-e u uslovne vektore koji će se koristiti za uslovljavanje generatora i diskriminatora
# tj omogućava GAN modelima da generišu slike koje su uslovljene na tekstualne opise!!!

class CANet(nn.Module):
  def __init__(self):
    super(CANet, self).__init__()
    self.text_dim = cfg.TEXT_DIMENSION
    self.c_dim = cfg.GAN_CONDITION_DIM
    self.fc = nn.Linear(self.text_dim, self.c_dim*2)
    self.relu = nn.ReLU()

  # encoding - transforms text embedding into a form that can be used to condition the GAN's generator and discriminator
  def encode(self, text_embedding):
    x = self.relu(self.fc(text_embedding))
    mu = x[:, :self.c_dim]
    logvar = x[:, self.c_dim:]
    return mu, logvar

# reparametrizacija - 
  def reparametrize(self, mu, logvar):
    std = logvar.mul(0.5).exp_()
    # eps - random noise
    eps = torch.randn(std.size(), device=mu.device)  # Ensuring eps is on the same device as mu
    return eps.mul(std).add_(mu)

  def forward(self, text_embedding):
    mu, logvar = self.encode(text_embedding)
    c = self.reparametrize(mu, logvar)
    return c, mu, logvar