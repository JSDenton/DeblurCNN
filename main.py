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
from DeblurCNN import DeblurCNN
from DSet import DeblurDataset
from yaml import load, dump
from sklearn.model_selection import train_test_split
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

#SETTINGS

sets = {} #settings (that's a global variable)
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
        if i=='desktop.ini': #pass the last item in folder (which is desktop.ini file)
            continue

        #img1 and img2 are first and one of the last images in the folder (that's specific for GOPRO dataset)

        #imgs_in_folder = os.listdir(f"{sets.get('dataset_path')}/{i}")        
        # img1 = cv2.imread(f"{sets.get('dataset_path')}/{i}/{imgs_in_folder[0]}", cv2.IMREAD_COLOR) 
        # img2 = cv2.imread(f"{sets.get('dataset_path')}/{i}/{imgs_in_folder[len(imgs_in_folder)-2]}", cv2.IMREAD_COLOR)

        img1 = cv2.imread(f"{sets.get('dataset_path')}/{i}", cv2.IMREAD_COLOR) 
        images.append(img1)
        #images.append(img2)
    print("The images have been read ...")
    return images

def gauss_blur(images):
    
    for f in os.listdir(sets.get('blurred_path')): #clear the folder
        os.remove(f"{sets.get('blurred_path')}/{f}")
    print("Cleared blurred images folder ...")
    os.makedirs(sets.get('blurred_path'), exist_ok = True)    
    for i in range(len(images)):
        images[i] = cv2.GaussianBlur(images[i], (31, 31), 0) #31 is a kernel size
        cv2.imwrite(f"{sets.get('blurred_path')}/img_{i}.png", images[i])
    print("The images have been blurred ...")
    return images

def validate(model, dataloader, val_data, device, criterion, epoch):
    model.eval()
    running_loss = 0.0
    i=0
    with torch.no_grad():
        for data in dataloader:
            blur_image = data[0]
            sharp_image = data[1]
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)
            outputs = model(blur_image)
            loss = criterion(outputs, sharp_image)
            running_loss += loss.item()
            if epoch == 0 and i == int((len(val_data)/dataloader.batch_size)-1):
                save_decoded_image(sharp_image.cpu().data, name=f"../outputs/saved_images/sharp{epoch}.jpg")
                save_decoded_image(blur_image.cpu().data, name=f"../outputs/saved_images/blur{epoch}.jpg")
            if i == int((len(val_data)/dataloader.batch_size)-1):
                save_decoded_image(outputs.cpu().data, name=f"../outputs/saved_images/val_deblurred{epoch}.jpg")
            i+=1

    return running_loss

def fit(model, dataloader, device, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        blur_image = data[0]
        sharp_image = data[1]
        blur_image = blur_image.to(device)
        sharp_image = sharp_image.to(device)
        optimizer.zero_grad()
        outputs = model(blur_image)
        loss = criterion(outputs, sharp_image)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        running_loss += loss.item()

    return running_loss

def __main__():
    if __name__!='__main__':
        return
    read_settings("settings.yaml")
    os.makedirs('../outputs/saved_images', exist_ok=True)
    for f in os.listdir(sets.get('blurred_path')): #clear the folder
        os.remove(f"{sets.get('blurred_path')}/{f}")
    imgs = get_images()
    blurr_imgs = gauss_blur(imgs)

    start_time = time.time()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((sets.get('image_size'), sets.get('image_size'))),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
    
    (x_train, x_val, y_train, y_val) = train_test_split(os.listdir(sets.get('blurred_path')), os.listdir(sets.get('dataset_path')), test_size=0.25)
    dataset = DeblurDataset(x_train, y_train, transform, settings=sets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=sets.get('batch_size'), shuffle = True, num_workers=sets.get('workers'))
    val_data = DeblurDataset(x_val, y_val, transform, settings=sets)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=sets.get('batch_size'), shuffle =False, num_workers=sets.get('workers'))


    device = torch.device("cuda:0" if (torch.cuda.is_available() and sets.get('ngpu') > 0) else "cpu") #using gpu (NVidia) for processing, otherwise using CPU

    model = DeblurCNN().to(device)
    #print(model)

    # the loss function
    criterion = nn.MSELoss()
    # the optimizer
    optimizer = optim.Adam(model.parameters(), lr=sets.get('learning_rate'))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        verbose=True
    )
    
    train_loss  = []
    val_loss = []  

    for epoch in range(sets.get('num_epochs')):
        print(f"Epoch {epoch+1} of {sets.get('num_epochs')}")
        train_epoch_loss = fit(model, dataloader, device, optimizer, criterion, epoch)
        val_epoch_loss = validate(model, dataloader, valloader, device, criterion, epoch)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        scheduler.step(val_epoch_loss)

    
    end_time = time.time()
    print(f"learning took {end_time - start_time} seconds")

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('../outputs/loss.png')
    plt.show()

    torch.save(model.state_dict(), '../outputs/model.pth')

__main__()