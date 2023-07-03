#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstrates facial landmark frontalization in a static image. Loads the image,
detects faces, extracts landmarks using DLIB and frontalizes them.

@author: Vasileios Vonikakis
"""
#%%
import yaml
import numpy as np
import dlib
import matplotlib.pyplot as plt
from utiles import frontalize_landmarks_dlib
import pandas as pd
from tqdm import tqdm
import imageio
from PIL import Image
path="/home/falhamdoosh/tgcn/data/PartA/vis/"
name_file = 'minidata'

config_file=open("./config/"+name_file+".yml", 'r')
config = yaml.safe_load(config_file)
data_path=config['data_path']
labels_path=config['labels_path']
edges_path=config['edges_path']
idx_train= config['idx_train']
idx_test=config['idx_test']
TS=config['TS']
batch_size=config['batch_size']
embed_dim=config['embed_dim']
num_features=config['num_features']
num_nodes=config['n_joints'] 
gpu=config['gpu']
name_exp=config['name_exp']
video_folder=config['video_path']
csv_file=config['csv_file']

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("/home/falhamdoosh/tgcn/facial_landmark_frontalization/data/shape_predictor_68_face_landmarks.dat") 
frontalization_weights = np.load('/home/falhamdoosh/tgcn/facial_landmark_frontalization/data/frontalization_weights.npy')

df = pd.read_csv(csv_file,sep='\t')
filesnames=(df['subject_name'] + '/' + df['sample_name']).to_numpy()
idx_train_data=np.load(idx_train)
filesnames_train=filesnames[idx_train_data]
label_data=np.load(labels_path)
label_train=label_data[idx_train_data]
#%%
for i in tqdm(range (0,len(filesnames_train))):
    frontalize_landmarks_dlib(video_folder,filesnames_train[i],detector,predictor,frontalization_weights)

# %%
data = np.zeros((20, 138, 68,2), dtype=np.float32)
landmarks_folder_frontalize="/home/falhamdoosh/tgcn/data/PartA/landmarks_frontalize/"
for i in tqdm(range (0,len(filesnames_train))):
    path=landmarks_folder_frontalize+filesnames_train[i]+".npy"
    sample=np.load(path) #[138,468,2] 
    data[i][:sample.shape[0]]=sample
#%%
path="/home/falhamdoosh/tgcn/data/PartA/vis/"
figures=[]
for time_step in tqdm(range(137)):
    # Create a figure and axes for subplots
    fig, axs = plt.subplots(5, 4, figsize=(30, 30))
    # Flatten the axes array
    axs = axs.flatten()
    for d in range(0,len(data)):
        fram=data[d,time_step,:,:]
        
        axs[d].scatter(fram[:,0], -fram[:,1], alpha=0.8)
        axs[d].set_title(f"VAS level: {label_train[d]}")
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    figures.append(image)
imageio.mimsave(path+'vis_landmarks_dlib.gif', figures, duration=50)
# %%
