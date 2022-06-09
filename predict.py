import sys
import numpy
import torch
import cv2
from DSet import DeblurDataset
from DeblurCNN import DeblurCNN
from matplotlib import pyplot as plt
from yaml import Loader, load
import torchvision.utils as vutils
import torchvision.transforms as transforms



def predict(args):
    # SPECIFY YOUR INPUT IMAGE PATH
    path = "./image.jpg"
    
    

    blur = True
    for arg in args:
        if arg.find("blur=")>-1:
            blur = bool(arg[5:])# 5 - jest długością "blur="
        elif arg.find("image_path=")>-1:
            path=arg[11:]
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    with open('settings.yaml', 'r') as file:
        sets = load(file, Loader = Loader) #read the settings yaml file
    if blur:
        image = cv2.GaussianBlur(image, (31, 31), 0)
    with torch.no_grad():
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((sets.get('image_size'), sets.get('image_size'))),
            transforms.ToTensor()
            ])

        model = DeblurCNN().to(torch.device('cpu'))
        model.load_state_dict(torch.load('../outputs/model.pth'))
        image = transform(image)
        model.eval()

        prediction = model(image)
        #image = numpy.argmax(prediction)
        img = prediction.cpu().data.view(prediction.cpu().data.size(0),  sets.get('image_size'), sets.get('image_size'))
        vutils.save_image(img, "./deblured_image.jpg")
        #image.reshape(sets.get('image_size'), sets.get('image_size'), 1)

        #cv2.imwrite("./deblured_img.jpg", image)
        # print(prediction.cpu().data)

        # plt.imshow(prediction.cpu().data)        
        # plt.show()        
        

predict(sys.argv)
# example run: py predict.py blur=False image_path='./img.png'