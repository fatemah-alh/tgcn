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

for i in tqdm(range (0,len(filesnames_train))):
    frontalize_landmarks_dlib(video_folder,filesnames_train[i],detector,predictor,frontalization_weights)

# %%
