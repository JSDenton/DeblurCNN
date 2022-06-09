from torch.utils.data import Dataset
import cv2

class DeblurDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths=None, transforms=None, settings=None):
        self.X = blur_paths 
        self.y = sharp_paths
        self.transforms = transforms
        self.settings = settings
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        #print(f"blur in Dset: {self.X[i]}")
        blur_image = cv2.imread(f"{self.settings.get('blurred_path')}/{self.X[i]}")
        
        if self.transforms:
            blur_image = self.transforms(blur_image)
            
        if self.y is not None:
            sharp_image = cv2.imread(f"{self.settings.get('dataset_path')}/{self.y[i]}")
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image