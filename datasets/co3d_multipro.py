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



def get_frame(sec, video):
  t_msec = 1000*sec
  video.set(cv2.CAP_PROP_POS_MSEC, t_msec) 
  ret, frame = video.read()
  height, width, _ = frame.shape
  frame = frame[: -int(0.2 * height), : , :]
  return frame


def find_patchdict(img1, img2):
  inliers, H = apply_sift(img1, img2) ## find homography
  if inliers == -1 or H is None:
    return None
      # row = {'folder': path, 'images' : f'({image_paths[i]}, {image_paths[j]} )', 'inliers': -1, '#matchedPatches': -1, 'percentage': 0}
      # df = df.append(row, ignore_index = True)
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
    return patchdict
      

def create_image2(sec, video):
  return cv2.resize(cv2.cvtColor(get_frame(sec, video), cv2.COLOR_BGR2RGB), (WIDTH, HEIGHT))

data_path = ""
store_path = ""

def main(folder):
  if os.path.exists(f"{store_path}/{folder}"):
    return
  os.mkdir(f"{store_path}/{folder}")
  image_paths = os.listdir(f"{data_path}/{folder}")
 
  image_paths.sort(key =lambda x : int(x[5:-4]))
  i = 0
  folder_counter = 0
  interval = 5
  while i < len(image_paths) - interval - 1:
      img1 =  cv2.resize(cv2.cvtColor(cv2.imread(f"{data_path}/{folder}/{image_paths[i]}"), cv2.COLOR_BGR2RGB), (WIDTH, HEIGHT))
      img2 = cv2.resize(cv2.cvtColor(cv2.imread(f"{data_path}/{folder}/{image_paths[i+ interval]}"), cv2.COLOR_BGR2RGB), (WIDTH, HEIGHT))
      patchdict = find_patchdict(img1, img2)
      if patchdict != None and len(patchdict)/N_TOTAL * 100 > lowerbound and  len(patchdict)/N_TOTAL * 100 < uperbound:
          os.mkdir(f"{store_path}/{folder}/{folder_counter}")
          cv2.imwrite(f"{store_path}/{folder}/{folder_counter}/0000.jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))  
          cv2.imwrite(f"{store_path}/{folder}/{folder_counter}/0001.jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))   
          save_corres(patchdict, f"{store_path}/{folder}/{folder_counter}")
          folder_counter += 1
      i+= (2* interval)
      print(colored(f'{folder} --- {folder_counter}', "cyan"))
        








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
  