import sys
import torch
import cv2
from DeblurCNN import DeblurCNN
from matplotlib import pyplot as plt
from yaml import Loader, load



def predict(args):
    # SPECIFY YOUR INPUT IMAGE PATH
    path = "./image.png"
    
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    print(args[1][5:]) # 5 - jest długością "blur="
    blur = True
    for arg in args:
        if arg.find("blur="):
            blur = bool(arg[5:])
        elif arg.find("image_path="):
            path=arg[11:]
    with open('settings.yaml', 'r') as file:
        sets = load(file, Loader = Loader) #read the settings yaml file
    cv2.resize(image, (sets.get('image_size'), sets.get('image_size')))
    if blur:
        cv2.GaussianBlur(image, (31, 31), 0)
    with torch.no_grad():
        model = DeblurCNN().to('cpu')
        model.load_state_dict(torch.load('../outputs/model.pth'))
        model.eval()

        prediction = model(image)

        cv2.imwrite("./deblured_img.jpg", image)

        plt.imshow(prediction)        
        plt.show()        
        

predict(sys.argv)
# example run: py predict.py blur=False image_path='./img.png'