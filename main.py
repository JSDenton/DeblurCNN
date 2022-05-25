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
import cv2
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

#SETTINGS

sets = {} #dictionary with a settings (that's a global variable)
img_size = 0

def read_settings(filename): #filename with its extension
    global sets
    with open(filename, 'r') as file:
        sets = load(file, Loader = Loader) #read the settings yaml file
        #print(type(settings))

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    vutils.save_image(img, name)

def get_images():
    path = os.listdir(sets.get('dataset_path'))
    images = []
    if len(path)<=0:
        print("WRONG PATH")
        return
    for i in path:
        imgs_in_folder = os.listdir(f"{sets.get('dataset_path')}/{i}")
        img1 = cv2.imread(imgs_in_folder[0], cv2.IMREAD_COLOR)
        img2 = cv2.imread(imgs_in_folder[len(imgs_in_folder)-2], cv2.IMREAD_COLOR)
        images.append(img1)
        images.append(img2)
    print("The images have been read ...")
    return images

def gauss_blur(images):
    os.makedirs('../blurred', exist_ok = True)
    for i in range(len(images)):
        images[i] = cv2.GaussianBlur(images[i], (31, 31), 0)
        cv2.imwrite(f"../blurred/{images[i]}", blur)

    return images

def __main__():
    if __name__!='__main__':
        return
    read_settings("settings.yaml")

    #print(settings)
    start_time = time.time()
    dataset = dset.ImageFolder(root = settings.get('dataset_path'),
                                                transform = transforms.Compose([
                                                    transforms.Resize(sets.get('image_size')),
                                                    transforms.CenterCrop(sets.get('image_size')),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                                ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = sets.get('batch_size'), shuffle = True, num_workers=sets.get('workers'))
    device = torch.device("cuda:0" if (torch.cuda.is_available() and sets.get('ngpu') > 0) else "cpu") #using gpu (NVidia) for processing, otherwise using CPU

__main__()