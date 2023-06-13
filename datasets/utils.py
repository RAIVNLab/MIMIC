import numpy as np
import cv2
import random
from constants import *


# by getting each point's (x,y) returns the index of the patch the point falls in
# indexing patches is left to right like: (for a 224x224 image with patch size = 16)
# 0  1  2  .... 13
# 14 15 16 ......
# .. .. .......195
def get_patch_index(x,y):
    assert x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT
    col = x // PATCH_SIZE
    row = y // PATCH_SIZE
    return row * NCOLS + col


## find the top left pixel coordinates of a patch
def get_xy_start(patch_idx):
    row = patch_idx // NCOLS 
    col = patch_idx % NCOLS
    x_start = col * PATCH_SIZE
    y_start = row * PATCH_SIZE
    return int(x_start), int(y_start)

## generate random pixel coordinates in a particular patch
def get_random_pixels_from_patch(patch_idx):
    randoms = np.array(random.choices(list(range(0,PATCH_SIZE * PATCH_SIZE)),k =NUM_PIXEL_RANDOM ))
    x_start, y_start = get_xy_start(patch_idx)
    xlist = randoms % PATCH_SIZE + x_start
    ylist = randoms // PATCH_SIZE + y_start
    return xlist,ylist


### checks how many patches contain inliers after key point matching
def find_num_inlier_patches(src_pts, matchesMask):
  patches = set()
  for i in range(len(matchesMask)):
    if matchesMask[i] == 1:
      xi, yi = src_pts[i,0,0], src_pts[i,0,1]
      patches.add(get_patch_index(xi, yi))
  return len(patches) 


# gets two images and return the homography transformation
def apply_sift(img1, img2):
  MIN_MATCH_COUNT = 10
  img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)      
  img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)      
  # sift key point detection
  sift = cv2.SIFT_create()
  kp1, des1 = sift.detectAndCompute(img1,None)
  kp2, des2 = sift.detectAndCompute(img2,None)
  if des1 is None or des2 is None or kp1 is None or kp2 is None:
    return -1, None
  # brute force matching
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
  good = []
  for i, pair in enumerate(matches):
    try:
        m, n = pair
        if m.distance < 0.7*n.distance:
            good.append(m)
    except ValueError:
        pass
  ## find homography
  if len(good) >  MIN_MATCH_COUNT:
      src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
      H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
      matchesMask = mask.ravel().tolist()
      return np.sum(matchesMask), H
  else:
      return -1, None

# gets a pixel (x,y) in the target image and homography transformation matrix 
# and returns the correspondence in the other image 
def find_match(H, x, y):
   pts = np.float32([ [x, y] ]).reshape(-1,1,2)
   dst = cv2.perspectiveTransform(pts,H) 
   dstx, dsty = dst[0,0,0], dst[0,0,1] 
   return dstx, dsty

# returns the most frequent element in a list with its occurance
def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])
    return maximum
     
# gets a patch index in the target image and homography matrix and returns the 
# corresponding patch index in the other image
def find_corres_patch(patchid, H):
    xs, ys = get_random_pixels_from_patch(patchid)
    all_matches = []
    for i in range(len(xs)):
        pts = np.float32([ [xs[i], ys[i]] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H) 
        dstx, dsty = dst[0,0,0], dst[0,0,1]
        if dstx < 0 or dsty < 0 or dstx >= WIDTH or dsty >= HEIGHT:
          all_matches.append("out")
          continue
        else:
            matchpatch = get_patch_index(int(dstx), int(dsty))
            all_matches.append(matchpatch)
    major = find_majority(all_matches)
    return major


## save metadata 
def save_corres(patchdict, path):
    np.save(f"{path}/corresponding.npy", patchdict)

