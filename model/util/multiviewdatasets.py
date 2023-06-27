# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae/
# DeiT: https://github.com/facebookresearch/deit

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from PIL import Image
import cv2
import numpy as np
import os
import random
from torchvision.transforms import transforms
import pandas as pd
import os
from torchvision import transforms, datasets
from termcolor import colored



class MultiViewDataset(Dataset):
    def __init__(self, csv_file, base_data_path, n_views, transform,  input_size, patch_size):
        self.paths = pd.read_csv(csv_file)
        self.n_views = n_views
        self.transform = transform
        self.BASE_PATH = base_data_path
        self.input_size = input_size
        print("input_size", input_size)
        print("patch_size", patch_size)
        self.lookup_table_size = int((input_size * input_size) / (patch_size * patch_size))

    def __getitem__(self, index):

        first_img_idx = random.choice([0,1])
        if first_img_idx == 0:
            img1 = self.transform(Image.open(os.path.join(self.BASE_PATH, self.paths['path1'].iloc[index])).convert("RGB"))
            img2 = self.transform(Image.open(os.path.join(self.BASE_PATH, self.paths['path2'].iloc[index])).convert("RGB"))
            # correspondences_dict = np.load(os.path.join(self.BASE_PATH + self.paths['correspondence'].iloc[index]), allow_pickle=True).tolist()
            # correspondences = [0] *  self.lookup_table_size
            # correspondence_mask = [0] * self.lookup_table_size
            # for key, val in correspondences_dict.items():
            #     correspondences[key] = val
            #     correspondence_mask[key] =1
            
        else:
            img1 = self.transform(Image.open(os.path.join(self.BASE_PATH, self.paths['path2'].iloc[index])).convert("RGB"))
            img2 = self.transform(Image.open(os.path.join(self.BASE_PATH, self.paths['path1'].iloc[index])).convert("RGB"))
            # correspondences_dict = np.load(os.path.join(self.BASE_PATH + self.paths['correspondence'].iloc[index]), allow_pickle=True).tolist()
            # correspondences = [0] *  self.lookup_table_size
            # correspondence_mask = [0] * self.lookup_table_size
            # # since the second img in the csv is the first image now
            # # correspondence mask is created to handle the case of zero 
            # for key, val in correspondences_dict.items():
            #     correspondences[val] = key
            #     correspondence_mask[val] =1

        return img1,img2 #, torch.tensor(correspondence_mask), torch.tensor(correspondences)


    def __len__(self):
        return len(self.paths) 

   

