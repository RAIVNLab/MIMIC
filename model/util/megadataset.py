import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from PIL import Image
import cv2
import numpy as np
import os
import random
import torchvision
from torchvision.transforms import transforms
import pandas as pd
import os


#from torchvision import transforms, datasets


class MegaDataset(Dataset):
    def __init__(self, csv_file, base_data_path, transforms):
        df = pd.read_csv(csv_file)

        
        path_view_1 = df['path1']
        path_view_2 = df['path2']

        

        df = pd.concat([path_view_1, path_view_2 ])
        print("final df shape", df.shape)

        self.paths = df
        self.base_path = base_data_path

        print("total paths ", len(self.paths) )
        self.transforms = transforms


    def __getitem__(self, index):
        img = self.transforms(Image.open(os.path.join(self.base_path, self.paths.iloc[index])).convert("RGB"))           
        return img, img # second variable is dummy  and not really used



    def __len__(self):
        return len(self.paths) 

   

