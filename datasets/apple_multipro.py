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
  t_msec = 1000*sec+1
  video.set(cv2.CAP_PROP_POS_MSEC, t_msec) 
  ret, frame = video.read()

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
      

def create_image(sec, video, degree):
  if degree == "w":
    return cv2.resize(cv2.cvtColor(get_frame(sec, video), cv2.COLOR_BGR2RGB), (WIDTH, HEIGHT))
  else:
    return cv2.rotate(cv2.resize(cv2.cvtColor(get_frame(sec, video), cv2.COLOR_BGR2RGB), (WIDTH, HEIGHT)), rotation_map[degree])






assert cv2.__version__ == "4.4.0"
rotation_map = { 'a':cv2.ROTATE_90_CLOCKWISE, 'z': cv2.ROTATE_180, 's':cv2.ROTATE_90_COUNTERCLOCKWISE}
data_path = ""
store_path = ""
dictionay = np.load("apple_rotation.npy", allow_pickle=True).item()


def main(folder):
  print(f'{data_path}/{folder}/{folder}.mov')
  if os.path.exists(f"{store_path}/{folder}") or folder not in dictionay:
    return
  os.mkdir(f"{store_path}/{folder}")
  video = cv2.VideoCapture(f'{data_path}/{folder}/{folder}.mov')
  fps = video.get(cv2.CAP_PROP_FPS)
  length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  seconds =  int(length // fps)
  i = 0
  if seconds < 5:
    return
  # print(colored(seconds, "red"))
  
  folder_counter = 0
  degree = dictionay[folder]
  print(colored(f"{i},  {folder},  {degree}", "green"))
  while i < seconds - 1:
    
    img1 =  create_image(i, video, degree)
    img2 = create_image(i+1 , video, degree)
    patchdict = find_patchdict(img1, img2)
    if patchdict != None and len(patchdict)/N_TOTAL * 100 > lowerbound and  len(patchdict)/N_TOTAL * 100 < uperbound:
      os.mkdir(f"{store_path}/{folder}/{folder_counter}")
      cv2.imwrite(f"{store_path}/{folder}/{folder_counter}/0000.jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))  
      cv2.imwrite(f"{store_path}/{folder}/{folder_counter}/0001.jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))   
      save_corres(patchdict, f"{store_path}/{folder}/{folder_counter}")
      folder_counter += 1
    elif patchdict != None and len(patchdict)/N_TOTAL * 100 > uperbound and i+2 < seconds:
      img2 = create_image(i+2 , video, degree)
      patchdict = find_patchdict(img1, img2)
      if patchdict != None and len(patchdict)/N_TOTAL * 100 > lowerbound and  len(patchdict)/N_TOTAL * 100 < uperbound:
        os.mkdir(f"{store_path}/{folder}/{folder_counter}")
        cv2.imwrite(f"{store_path}/{folder}/{folder_counter}/0000.jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))  
        cv2.imwrite(f"{store_path}/{folder}/{folder_counter}/0001.jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))   
        save_corres(patchdict, f"{store_path}/{folder}/{folder_counter}")
        folder_counter += 1
        i += 1
    i+=1
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
  