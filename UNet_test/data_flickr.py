import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch






class FlickrBase(Dataset):
    def __init__(self,name, text_path, vision_path):
        self.name = name
        self.text_embeds = torch.load(text_path)
        self.vision_embeds = torch.load(vision_path)
        # self.original_image = torch.load(original_image_path)


    def __getitem__(self, index: int):
        data = dict()
        # data["original_image"] = self.vision_embeds[index]# self.original_image[index]
        data["vision_embedding"] = self.vision_embeds[index]
        data["text_embedding"] = self.text_embeds[index]
        return data

    def __len__(self) -> int:
        return self.vision_embeds.shape[0]
    
    



class FlickrTrain100(FlickrBase):
    def __init__(self, text_path, vision_path):
        super().__init__(name="Train100",
                        text_path=text_path,
                         vision_path=vision_path)
        
        
class FlickrValidation100(FlickrBase):
     def __init__(self, text_path, vision_path):
        super().__init__(name="val100",
                        text_path=text_path,
                         vision_path=vision_path)

class FlickrTest100(FlickrBase):
     def __init__(self, text_path, vision_path):
        super().__init__(name="Test100",
                        text_path=text_path,
                         vision_path=vision_path)
        

class FlickrTrain(FlickrBase):
    def __init__(self, text_path, vision_path):
        super().__init__(name="Train",
                        text_path=text_path,
                         vision_path=vision_path)
 
        
        
class FlickrValidation(FlickrBase):
     def __init__(self, text_path, vision_path):
        super().__init__(name="val",
                        text_path=text_path,
                         vision_path=vision_path)
        
class FlickrTest(FlickrBase):
     def __init__(self, text_path, vision_path):
        super().__init__(name="test",
                        text_path=text_path,
                         vision_path=vision_path)
        


