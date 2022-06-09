from types import NoneType
from torch.utils.data import Dataset
import cv2

class DeblurDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths=None, transforms=None, settings=None):
        # ścieżki do obrazków
        self.X = blur_paths 
        self.y = sharp_paths

        #transformata
        self.transforms = transforms

        # ustawienia (żeby móc zachować głównie rozdzielczość, a nie wpisywać "z ręki")
        self.settings = settings
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        blur_path = ""
        sharp_path = ""
        # sprawdzenie, czy ustawienia istnieją, i dostosowanie ścieżek
        if type(self.settings) is not NoneType:            
            blur_path = self.settings.get('blurred_path')
            sharp_path = self.settings.get('dataset_path')
        blur_image = cv2.imread(f"{blur_path}/{self.X[i]}")
        
        # zaaplikowanie transformaty
        if self.transforms:
            # zamiana kolorów z BGR na RGB (patrz: sekcja Wnioski -> Problemy w trakcie implementacji)
            blur_image = blur_image[:,:,::-1]
            blur_image = self.transforms(blur_image)
                
        if self.y is not None:
            sharp_image = cv2.imread(f"{sharp_path}/{self.y[i]}")
            sharp_image = sharp_image[:,:,::-1]
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image