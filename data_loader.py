import os
import numpy as np

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from utils import grayscale_loader

def get_train_loader(root_dir, 
                     transform, 
                     batch_size=128,
                     shuffle=True,
                     num_workers=4):

    train_data = ImageFolder(root=root_dir, 
                             loader=grayscale_loader, 
                             transform=transform)

    train_loader = DataLoader(train_data, 
                              batch_size=128, 
                              shuffle=True, 
                              num_workers=4)

    return train_loader