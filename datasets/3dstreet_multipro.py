### imports 

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import random
import cv2
import warnings
import argparse
from tqdm import tqdm
warnings.filterwarnings("ignore")
from termcolor import colored
from multiprocessing import Pool
from utils import *
from constants import *

data_path = ""
store_path = ""

def main(folder):
  print(colored(folder, "cyan"))
  image_paths = [f  for f in os.listdir(os.path.join(data_path, folder)) if f.endswith("jpg")]
  if os.path.exists(f"{store_path}/{folder}") or len(image_paths) < 2:
    return
  os.mkdir(f"{store_path}/{folder}")
  counter = 0
  for i  in range(len(image_paths)):
    for j in range(i+1,len(image_paths)):
      counter +=1
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
        # print(len(patchdict1) ,len(patchdict2))
        
        if len(patchdict1) < len(patchdict2) :
          patchdict = patchdict1
        else:
          patchdict =  {v: k for k, v in patchdict2.items()}
        # print(colored(len(patchdict)/N_TOTAL * 100,"green"))
        if len(patchdict)/N_TOTAL * 100 >= lowerbound and  len(patchdict)/N_TOTAL * 100 <= uperbound:
          # print(f"{image_paths[i]} -- {image_paths[j]}", len(patchdict)/N_TOTAL * 100 )
          os.system(f"mkdir {store_path}/{folder}/{counter}")
          cv2.imwrite(f"{store_path}/{folder}/{counter}/0000.jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))  
          cv2.imwrite(f"{store_path}/{folder}/{counter}/0001.jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))   
          save_corres(patchdict, f"{store_path}/{folder}/{counter}")
          assert len(patchdict) == len(np.unique(list(patchdict.values())))
  



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
  