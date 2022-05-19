import argparse
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

#SETTINGS

settings = {} #dictionary with a settings (that's a global variable)
img_size = 0

def read_settings(filename): #filename with its extension
    global settings
    with open(filename, 'r') as file:
        settings = load(file, Loader = Loader) #read the settings yaml file
        #print(type(settings))
                


def __main__():
    if __name__!='__main__':
        return
    read_settings("settings.yaml")
    #print(settings)
    start_time = time.time()
    dataset = dset.ImageFolder(root = settings.get('dataset_path'),
                                                transform = transforms.Compose([
                                                    transforms.Resize(settings.get('image_size')),
                                                    transforms.CenterCrop(settings.get('image_size')),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                                ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = settings.get('batch_size'), shuffle = True, num_workers=settings.get('workers'))
    device = torch.device("cuda:0" if (torch.cuda.is_available() and settings.get('ngpu') > 0) else "cpu") #using gpu (NVidia) for processing, otherwise using CPU

__main__()