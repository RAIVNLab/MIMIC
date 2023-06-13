# Dataset Curation

* In the `dataset` folder, there is a separate python script for each dataset such as Mannequin, HM3D and .... 

* In all scripts, we use the multiprocessing package in Python to make the process faster. We pass each scene for 3Dscenes datasets or each video in video datasets to a single process. 

* Each script gets the path to the raw data, mines potential image pairs, which is different for videos and 3D scenes (.glb files), and then stores the image pairs with sufficient overlap.

* All you need to do is download your raw dataset (for example, co3d) and use the command below: (`data_path` points to the raw data directory, which has all scenes/videos)
```bash
python co3d_multipro.py --data_path /path/to/raw/files --store_path /path/to/store/directory --cpus num_cpus
```

* You can easily add a new dataset using the same procedure in the provided scripts. For video datasets, you need to set the interval in which the potential frames are collected that depends on the speed of the video in capturing the environment. For 3D scenes, you must set the maximum number of image pairs you want to store from a single .glb file.


* For running codes that need the habitat simulator, you need to have a GPU, but for others CPUs are enough

* For installing habitat simulator and understanding the code for [HM3D](hm3d_multipro.py), [Gibson](gibson_multipro.py) and [Matterport](matterport_multipro.py) better, you can see this [repo](https://github.com/facebookresearch/habitat-sim).

* Raw scenes or videos of each dataset can be found in the below links:
    - DeMoN: https://github.com/lmb-freiburg/demon/tree/master
    - ScanNet: https://github.com/ScanNet/ScanNet
    - HM3D: https://matterport.com/partners/facebook
    - Gibson:https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md
    - Matterport: https://github.com/niessner/Matterport/tree/master
    - 3DStreetView: https://github.com/amir32002/3D_Street_View
    - CO3D: https://github.com/facebookresearch/co3d/tree/main
    - Objectron: https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Download%20Data.ipynb
    - Mannequin: https://google.github.io/mannequinchallenge/www/download.html
    - ArkitScenes: https://github.com/apple/ARKitScenes/blob/main/DATA.md