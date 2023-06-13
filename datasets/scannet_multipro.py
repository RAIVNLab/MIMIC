### imports 

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import random
import cv2
import pandas as pd
import warnings
import argparse
from tqdm import tqdm
warnings.filterwarnings("ignore")
from termcolor import colored
from multiprocessing import Pool
from constants import *
from utils import *



data_path = ""
store_path = ""

def main(folder):
  print(colored(folder, "cyan"))
  if os.path.exists(f"{store_path}/{folder}"):
    return
  os.mkdir(f"{store_path}/{folder}")
  image_paths = [f  for f in os.listdir(os.path.join(data_path, folder)) if f.endswith("jpg")]
  NUM_PAIRS =  100
  NUM_Compares = 10
  folder_counter = 0
  STEP =  60
  NUM_PAIRS =  len(image_paths) // STEP
  print(colored(STEP, "magenta"))
  indexes = list(range(0,len(image_paths) - STEP, STEP))
  print(colored(indexes, "blue"))
  for index in indexes:
    start_idx = list(range(index, index + NUM_Compares))
    end_idx = list(range(index + STEP - 1 - NUM_Compares, index + STEP - 1))

  # print(image_paths)
    best_percent = 100
    best_dict = None
    pair = [None, None]
    for i  in start_idx:
      for j in end_idx:
        PATH1 = os.path.join(data_path, folder, image_paths[i])
        PATH2 = os.path.join(data_path, folder, image_paths[j])
        img1 = cv2.resize(cv2.cvtColor(cv2.imread(PATH1), cv2.COLOR_BGR2RGB), (WIDTH, HEIGHT))
        img2 = cv2.resize(cv2.cvtColor(cv2.imread(PATH2), cv2.COLOR_BGR2RGB), (WIDTH, HEIGHT))
        inliers, H = apply_sift(img1, img2) ## find homography
        if inliers == -1 or H is None:
          pass
        else :
          patchdict1 = {}
          for patchid in range(0, NCOLS * NROWS):
            corres_patch , _ = find_corres_patch(patchid, H)
            if corres_patch != 'out' and corres_patch not in list(patchdict1.values()):
              patchdict1[patchid] = corres_patch

          patchdict2 = {}
          for patchid in range(0, NCOLS * NROWS):
            corres_patch , _ = find_corres_patch(patchid, np.linalg.inv(H))
            if corres_patch != 'out' and corres_patch not in list(patchdict2.values()):
              patchdict2[patchid] = corres_patch
          
          if len(patchdict1) < len(patchdict2) :
            patchdict = patchdict1
          else:
            patchdict =  {v: k for k, v in patchdict2.items()}
          
          if len(patchdict)/N_TOTAL * 100 > lowerbound and  len(patchdict)/N_TOTAL * 100 < uperbound:
            if len(patchdict)/N_TOTAL * 100 < best_percent:
              best_percent = len(patchdict)/N_TOTAL * 100
              best_dict = patchdict
              pair = [img1, img2]
            assert len(patchdict) == len(np.unique(list(patchdict.values())))

    print(colored(best_percent , "green"))
    if best_percent < 100:
      os.mkdir(f"{store_path}/{folder}/{folder_counter}")
      assert best_dict != None
      assert best_percent > 50 
      cv2.imwrite(f"{store_path}/{folder}/{folder_counter}/0000.jpg", cv2.cvtColor(pair[0], cv2.COLOR_RGB2BGR))  
      cv2.imwrite(f"{store_path}/{folder}/{folder_counter}/0001.jpg", cv2.cvtColor(pair[1], cv2.COLOR_RGB2BGR))   
      save_corres(best_dict, f"{store_path}/{folder}/{folder_counter}")
      folder_counter += 1

 


if __name__ == "__main__":
    
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", type=str)
  parser.add_argument("--store_path",type=str )
  parser.add_argument("--cpus",type=int )
  args = parser.parse_args()
  data_path = args.data_path
  store_path = args.store_path
  folders = os.listdir(args.data_path)
  p = Pool(args.cpus)
  p.map(main, folders)
  p.close()
  p.join()
  