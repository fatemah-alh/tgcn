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
predictor = dlib.shape_predictor("/home/falhamdoosh/facial_landmark_frontalization/data/shape_predictor_68_face_landmarks.dat") 
frontalization_weights = np.load('/home/falhamdoosh/facial_landmark_frontalization/data/frontalization_weights.npy')

#Read 20 sample
df = pd.read_csv(csv_file,sep='\t')
filesnames=(df['subject_name'] + '/' + df['sample_name']).to_numpy()
idx_train_data=np.load(idx_train)
filesnames_train=filesnames[idx_train_data]
label_data=np.load(labels_path)
label_train=label_data[idx_train_data]
#%%
#using EOS model
for i in tqdm(range (0,len(filesnames_train))):
    frontalize_landmarks_dlib(video_folder,filesnames_train[i],detector,predictor,frontalization_weights)

# %%
data = np.zeros((20, 138, 68,2), dtype=np.float32)
#landmarks_folder_frontalize="/home/falhamdoosh/tgcn/data/PartA/landmarks_frontalize/"
landmarks_folder="/home/falhamdoosh/tgcn/data/PartA/3Dlandmarks/"
for i in tqdm(range (0,len(filesnames_train))):
    path=landmarks_folder+filesnames_train[i]+"2d_landmarks.npy"
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
imageio.mimsave(path+'vis_landmarks_3D.gif', figures, duration=50)
#%%
#3D visualizzation
data_3d = np.zeros((20, 138, 68,3), dtype=np.float32)
#landmarks_folder_frontalize="/home/falhamdoosh/tgcn/data/PartA/landmarks_frontalize/"
landmarks_folder="/home/falhamdoosh/tgcn/data/PartA/3Dlandmarks/"
for i in tqdm(range (0,len(filesnames_train))):
    path=landmarks_folder+filesnames_train[i]+"3d_landmarks.npy"
    sample=np.load(path) #[138,468,2] 
    data_3d[i][:sample.shape[0]]=sample
#%%
path="/home/falhamdoosh/tgcn/data/PartA/vis/"
figures=[]
for time_step in tqdm(range(137)):
    # Create a figure and axes for subplots
    fig, axs = plt.subplots(5, 4, figsize=(30, 30),subplot_kw=dict(projection='3d'))
   # fig = plt.figure(figsize=(30, 30))

    # Flatten the axes array
    axs = axs.flatten()
    for d in range(0,len(data_3d)):
        fram=data_3d[d,time_step,:,:]
        
        axs[d].scatter(fram[:,0], fram[:,1],fram[:,2], alpha=0.8)
        axs[d].set_title(f"VAS level: {label_train[d]}")
        axs[d].view_init(120,-120)
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    figures.append(image)
    
imageio.mimsave(path+'vis_landmarks_3D_3D.gif', figures, duration=50)


# %%

#Align points to autovectors.
from utiles import standardize,frobenius_norm

import cv2
from numpy.linalg import eig

def get_eigenvectors(arr):
    cova=np.cov(arr,rowvar=False)
    eigenvalues, eigenvectors = eig(cova)
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
   # eigenvectors=np.absolute(eigenvectors)
    print(eigenvectors)
    return eigenvectors

def align_eigenvectors(eigenvectors,arr):
    arr=np.dot(arr, eigenvectors.T) # 68*3 __ 3*3 
    return arr
def preprocess_frame(sample):
    sample= standardize(sample)
    sample=frobenius_norm(sample)
    return sample
path="/home/falhamdoosh/tgcn/data/PartA/vis/"
figures=[]
#writer = imageio.get_writer('test.mp4')
origin=[0,0,0]
vis="2d"
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
        #fram=np.concatenate( (data[d,time_step,:,0:1],-data[d,time_step,:,1:2],data_3d[d,time_step,:,2:]),axis=1 )
        fram=preprocess_frame(fram)
        autovettori=get_eigenvectors(fram)
        #fram=align_eigenvectors(autovettori,fram)
        #autovettori=get_eigenvectors(fram)
        axs[d].set_title(f"VAS level: {label_train[d]}")
        if vis=="2d":
            axs[d].scatter(fram[:,0], fram[:,1], alpha=0.8)
        else:
            axs[d].scatter(fram[:,0], fram[:,1],fram[:,2], alpha=0.8)
            axs[d].quiver(origin[0],origin[1],origin[2], autovettori[0,0],autovettori[1,0],autovettori[2,0],color="r")
            axs[d].quiver(origin[0],origin[1],origin[2], autovettori[0,1],autovettori[1,1],autovettori[2,1],color="g")
            axs[d].quiver(origin[0],origin[1],origin[2], autovettori[0,2],autovettori[1,2],autovettori[2,2],color="b")
            axs[d].quiver(origin[0],origin[1],origin[2],0,0,1,color="y")
            axs[d].text(0,0,1,"z")
            axs[d].quiver(origin[0],origin[1],origin[2], 0,1,0,color="y")
            axs[d].text(0,1,0,"y")
            axs[d].quiver(origin[0],origin[1],origin[2], 1,0,0,color="y")
            axs[d].text(1,0,0,"x")
            #axs[d].view_init(120,-120)
        
    fig.canvas.draw()
    
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    figures.append(image)


imageio.mimsave(path+'vis_landmarks_2D_from_3D_model.gif', figures, duration=50)


# %%
