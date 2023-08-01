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




#%%
figures=[]
#writer = imageio.get_writer('test.mp4')
origin=[0,0,0]
vis="2d"
vis_indix=False
for time_step in tqdm(range(137)):
    # Create a figure and axes for subplots
    if vis=="2d":
        fig, axs = plt.subplots(5, 4, figsize=(30, 30))
    else:
        fig, axs = plt.subplots(5, 4, figsize=(30, 30),subplot_kw=dict(projection='3d'))
    
    # Flatten the axes array
    axs = axs.flatten()
    for d in range(0,len(data_3d)):
        
        fram=data_3d[d,time_step,:,:]
        fram[:,1]=-fram[:,1]
        
        
        fram=preprocess_frame(fram)
       
        base_m = get_rotation_matrix(fram)
        fram= np.matmul( base_m, fram.T ).T
        
        axs[d].set_title(f"VAS level: {label_data[d]}")
        if vis_indix:
            for index in range(len(fram)):
                axs[d].annotate(index,(fram[index][0],fram[index][1]))
        if vis=="2d":
            axs[d].scatter(fram[:,0], fram[:,1], alpha=0.8)
        else:
            axs[d].scatter(fram[:,0], fram[:,1],fram[:,2], alpha=0.8)
           # axs[d].quiver(origin[0],origin[1],origin[2], autovettori[0,0],autovettori[1,0],autovettori[2,0],color="r")
            #axs[d].quiver(origin[0],origin[1],origin[2], autovettori[0,1],autovettori[1,1],autovettori[2,1],color="g")
            #axs[d].quiver(origin[0],origin[1],origin[2], autovettori[0,2],autovettori[1,2],autovettori[2,2],color="b")
            #axs[d].quiver(origin[0],origin[1],origin[2], n[0],n[1],n[2],color="fuchsia")
            axs[d].quiver(origin[0],origin[1],origin[2],0,0,1,color="y")
            axs[d].text(0,0,1,"z")
            axs[d].quiver(origin[0],origin[1],origin[2], 0,1,0,color="y")
            axs[d].text(0,1,0,"y")
            axs[d].quiver(origin[0],origin[1],origin[2], 1,0,0,color="y")
            axs[d].text(1,0,0,"x")
            #axs[d].view_init(-30,60)
        
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    figures.append(image)

#imageio.mimsave(path_vis+'vis_landmarks_3D_openface_absolut_eigenvectors_without_contur.gif', figures, duration=50)
#%%