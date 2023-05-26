#%%
import matplotlib.pyplot as plt
import cv2
from mlxtend.image import extract_face_landmarks
import numpy as np
from scipy.spatial import Delaunay
import pandas as pd
from tqdm import tqdm 
import torch
import os,sys
import dlib
import mediapipe as mp
import math
import pickle
import yaml
from yaml import FullLoader
from PIL import Image
import imageio
from utiles import  get_mini_dataset,extract_landmarks_from_video_media_pipe,calc_velocity,standardize,get_idx_train_test,extract_landmarks_from_video
from dataloader import DataLoader

path="/home/falhamdoosh/tgcn/data/PartA/minidata/vis"
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

train_dataset=DataLoader(data_path,labels_path,edges_path,name_exp,idx_path=idx_train,mode="train")
#test_dataset=DataLoader(data_path,labels_path,edges_path,name_exp,idx_path=idx_test,mode="test")
writer = imageio.get_writer(path+'/vis_landmarks_mediapipe.mp4')

figures = []
for time_step in tqdm(range(137)):
    # Create a figure and axes for subplots
    fig, axs = plt.subplots(5, 4, figsize=(25, 25))
    # Flatten the axes array
    axs = axs.flatten()
    for i,sample in enumerate(train_dataset):
        x,y,_,_=sample
        axs[i].scatter(x[:,0,time_step], -x[:,1,time_step], alpha=0.8)
        axs[i].set_title(f"VAS level: {y}")
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    figures.append(image)
    writer.append_data(image)
writer.close()
    #plt.show()
#%%
#imageio.mimsave(path+'/vis_landmarks_mediapipe.gif', figures, duration=50)