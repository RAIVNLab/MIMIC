import math
import os
import random
import sys
import glob
import git
import imageio
import magnum as mn
import numpy as np
import cv2 as cv
import argparse

from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

import os
import random
import cv2
import pandas as pd
from constants import *

from syn_utils import *
from utils import *
from termcolor import colored

from multiprocessing import Pool


data_path = ""
store_path = ""

def main(folder):

    if os.path.exists(f"{store_path}/{folder}"):
        return
    print(colored(folder, "magenta"))
    os.mkdir(f"{store_path}/{folder}")
    for f in os.listdir(os.path.join(data_path, folder)):
        if f.endswith(".basis.glb"):
            scene = os.path.join(data_path, folder, f)
            break
    rgb_sensor = True 
    depth_sensor = False  
    semantic_sensor = False 
    

    sim_settings = {
        "width": 256,  # Spatial resolution of the observations
        "height": 256,
        "scene": scene,  # Scene path
        "default_agent": 0,
        "sensor_height": random.uniform(1.2, 1.8),  # Height of sensors in meters
        "color_sensor": rgb_sensor,  # RGB sensor
        "depth_sensor": depth_sensor,  # Depth sensor
        "semantic_sensor": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
    }


    cfg = make_cfg(sim_settings)
    # Needed to handle out of order cell run in Colab
    try:  # Got to make initialization idiot proof
        sim.close()
    except NameError:
        pass
    sim = habitat_sim.Simulator(cfg)


    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    


    num_frames = 360 // DEFAULT_DEGREE
    num_samples = 3000
    sample = 0
    while sample < num_samples:
        
        A = []
        B = []
        C = []
        nav_point = sim.pathfinder.get_random_navigable_point()
        # nav_point = sim.pathfinder.get_random_navigable_point()
        # nav_point = np.array([-2.1291089e+00, -5.3811073e-04, -2.7964590e+00])
        # print(nav_point)
        if sim.pathfinder.island_radius(nav_point) < 6 :
            print("small island")
            sample += 1
            continue
        agent = sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(nav_point)  
        agent.set_state(agent_state)

        set_default(sim)
        assert sim.config.agents[0].action_space["turn_right"].actuation.amount == DEFAULT_DEGREE
        assert sim.config.agents[0].action_space["move_forward"].actuation.amount == DEFAULT_STEP
        for frame in range(num_frames):
            action = "turn_right"
            observations = sim.step(action)
            rgb = observations["color_sensor"]
            rgb = np.array(Image.fromarray(rgb[:,:,:3], mode="RGB"))
            rgb = crop_center(rgb, WIDTH, HEIGHT)
            assert rgb.shape == (WIDTH, HEIGHT, 3)
            A.append(rgb)
        A = [A[-1]] + A[:-1]
        assert len(A) == 360 // DEFAULT_DEGREE
        # show_images(A)


        ### how much turn? (dont choose the 0 and 360 degree)
        degree = random.choice([1,2,4,5]) * 60
        sim.config.agents[0].action_space['turn_right']= habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=degree))
        assert sim.config.agents[0].action_space["turn_right"].actuation.amount == degree
        observations = sim.step("turn_right")
        step = random.uniform(0.5, 1)
        sim.config.agents[0].action_space['move_forward']= habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=step))
        assert sim.config.agents[0].action_space["move_forward"].actuation.amount == step
        observations = sim.step("move_forward")




        set_default(sim)
        assert sim.config.agents[0].action_space["turn_right"].actuation.amount == DEFAULT_DEGREE
        assert sim.config.agents[0].action_space["move_forward"].actuation.amount == DEFAULT_STEP
        
        for frame in range(num_frames):
            action = "turn_right"
            observations = sim.step(action)
            rgb = observations["color_sensor"]
            rgb = np.array(Image.fromarray(rgb[:,:,:3], mode="RGB"))
            rgb = crop_center(rgb, WIDTH, HEIGHT)
            assert rgb.shape == (WIDTH, HEIGHT, 3)
            B.append(rgb)
        B = [B[-1]] + B[:-1]
        assert len(B) == 360 // DEFAULT_DEGREE
        # show_images(B)


        
        ### in the other sets 180 degree is not allowed
        degree = random.choice([0,1,2,4,5]) * 60
        sim.config.agents[0].action_space['turn_right']= habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=degree))
        observations = sim.step("turn_right")
        step = random.uniform(0.5, 1)
        sim.config.agents[0].action_space['move_forward']= habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=step))
        observations = sim.step("move_forward")

    
        set_default(sim)
        assert sim.config.agents[0].action_space["turn_right"].actuation.amount == DEFAULT_DEGREE
        assert sim.config.agents[0].action_space["move_forward"].actuation.amount == DEFAULT_STEP
        for frame in range(num_frames):
            action = "turn_right"
            observations = sim.step(action)
            rgb = observations["color_sensor"]
            rgb = np.array(Image.fromarray(rgb[:,:,:3], mode="RGB"))
            rgb = crop_center(rgb, WIDTH, HEIGHT)
            assert rgb.shape == (WIDTH, HEIGHT, 3)
            C.append(rgb)
        C = [C[-1]] + C[:-1]
        assert len(C) == 360 // DEFAULT_DEGREE
        # show_images(C)
        



        best_percent = 100
        best_dict = None
        BorC = None
        ids = ()
        for ida, a in enumerate(A):
            if cv2.Laplacian(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() < VARIANCE_THRESHOLD:
                continue
            for idb, b in enumerate(B):
                if cv2.Laplacian(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() < VARIANCE_THRESHOLD:
                    continue
                is_good, percent, patchdict = overlap(a,b, "")
                if is_good:
                    if percent < best_percent:
                        best_percent = percent
                        best_dict = patchdict
                        BorC = "B"
                        ids = (ida, idb)
            for idc, c in enumerate(C):
                if cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() < VARIANCE_THRESHOLD:
                    continue
                is_good, percent, patchdict = overlap(a, c, "")
                if is_good:
                    if percent < best_percent:
                        best_percent = percent
                        best_dict = patchdict
                        BorC = "C"
                        ids = (ida, idc)
        
        if best_percent < 100:
            newpath = f"{store_path}/{folder}/{str(sample)}"
            os.mkdir(newpath)
            cv2.imwrite(f"{newpath}/0000.jpg", cv2.cvtColor(A[ids[0]], cv2.COLOR_RGB2BGR ))
            save_corres(best_dict, newpath)
            if BorC == "B":
                cv2.imwrite(f"{newpath}/0001.jpg", cv2.cvtColor(B[ids[1]], cv2.COLOR_RGB2BGR ))
            else:
                cv2.imwrite(f"{newpath}/0001.jpg", cv2.cvtColor(C[ids[1]], cv2.COLOR_RGB2BGR ))
        print(colored(f"{folder}--{str(sample)}", "green"))
        sample+=1


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