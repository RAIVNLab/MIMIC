
import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from utils import *
from constants import *

DEFAULT_DEGREE = 45
DEFAULT_STEP = 0.5
VARIANCE_THRESHOLD = 1000

def set_default(sim):
  sim.config.agents[0].action_space['turn_right']= habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=DEFAULT_DEGREE))
  sim.config.agents[0].action_space['turn_left']= habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=DEFAULT_DEGREE))
  sim.config.agents[0].action_space['move_forward']= habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=DEFAULT_STEP))

def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, len(arr), i + 1)
        ax.axis("off")
        plt.imshow(data)
    plt.show(block=False)

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.5)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=45.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=45.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])




def crop_center(img,cropx,cropy):
    y,x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]


# getting two images returns if they have sufficient overlap or not, if they had it would return the 
# percentage and matching dictionary too
def overlap(img1, img2, name):
    img1 = cv2.resize(img1, (WIDTH, HEIGHT))
    img2 = cv2.resize(img2, (WIDTH, HEIGHT))
    inliers, H = apply_sift(img1, img2) ## find homography
    if inliers == -1 or H is None:
        return False, None, None     
    else :
        ## matching image 1 - > image 2
        patchdict1 = {}
        for patchid in range(0, NCOLS * NROWS):
            corres_patch , _ = find_corres_patch(patchid, H)
            if corres_patch != 'out' and corres_patch not in list(patchdict1.values()):
                patchdict1[patchid] = corres_patch
        ## matching image 2 - > image 1
        patchdict2 = {}
        for patchid in range(0, NCOLS * NROWS):
            corres_patch , _ = find_corres_patch(patchid, np.linalg.inv(H))
            # only count the patches that havent been matched with previous patches and are not outside the image boundaries
            if corres_patch != 'out' and corres_patch not in list(patchdict2.values()):
                patchdict2[patchid] = corres_patch
        if len(patchdict1) < len(patchdict2) :
          patchdict = patchdict1
        else:
          ## change the matching to image 1 -> image 2
          patchdict =  {v: k for k, v in patchdict2.items()}
          assert len(patchdict) == len(patchdict2)
        if len(patchdict)/N_TOTAL * 100 > lowerbound and  len(patchdict)/N_TOTAL * 100 < uperbound:
            return True, len(patchdict)/N_TOTAL * 100, patchdict

        assert len(patchdict) == len(np.unique(list(patchdict.values())))
    return False, None, None
  
