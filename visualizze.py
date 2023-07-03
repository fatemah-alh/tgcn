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
from utiles import align_points,align_points_degree,align_covariance
from dataloader import DataLoader


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
#eyes=[384, 385, 386, 387, 388, 133, 390, 263, 7, 398, 144, 145, 153, 154, 155, 157, 158, 159, 160, 33, 161, 163, 173, 466, 362, 373, 246, 374, 249, 380, 381, 382]
eyes=[263,33]
train_dataset=DataLoader(data_path,labels_path,edges_path,name_exp,idx_path=idx_train,mode="train")
#test_dataset=DataLoader(data_path,labels_path,edges_path,name_exp,idx_path=idx_test,mode="test")
figures = []
for time_step in tqdm(range(137)):
    # Create a figure and axes for subplots
    fig, axs = plt.subplots(5, 4, figsize=(30, 30))
    # Flatten the axes array
    axs = axs.flatten()
    for i,sample in enumerate(train_dataset):
        x,y,_,_=sample
        fram=x[:,:2,time_step]
        #autovettori=align_covariance(fram)
       # fram=autovettori[:,0:2]
       # print(fram.shape) 
       # frame=rotated_points=np.dot(rotated_points,[[-1,0],[0,-1]].T) # 180
       # fram=align_points_degree(fram)   
        #fram=align_points(fram)   
        #print(fram.shape)     
        axs[i].scatter(fram[:,0], fram[:,1], alpha=0.8)
        #annotate points:
       # for kk in range(len(eyes)):
          #  index=eyes[kk]
          #  axs[i].annotate(index,(fram[index,0],fram[index,1]))
        axs[i].set_title(f"VAS level: {y}")
        
    
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    figures.append(image)
    
   
    
   

imageio.mimsave(path+'vis_landmarks_dlib.gif', figures, duration=50)

# %%
