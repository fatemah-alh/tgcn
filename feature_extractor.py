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

from utiles import  get_mini_dataset,extract_landmarks_from_video_media_pipe,calc_velocity,standardize,get_idx_train_test,extract_landmarks_from_video



name_exp = 'dlib'
config_file=open("./config/"+name_exp+".yml", 'r')
config = yaml.safe_load(config_file)
if torch.cuda.is_available():
  torch.cuda.set_device(0)
  device = 'cuda'  
else:
  device='cpu'
dlib.cuda.set_device(0)
#%%
#Create Data matrix and normalize..
landmarks_folder="/home/falhamdoosh/tgcn/data/PartA/landmarks/"
video_folder="/home/falhamdoosh/tgcn/data/PartA/"
df = pd.read_csv('/home/falhamdoosh/tgcn/data/PartA/samples.csv',sep='\t')
labels=df['class_id'].values
filesnames=(df['subject_name'] + '/' + df['sample_name']).to_numpy()

normalized_data = np.zeros((8700, 137, 469, 4), dtype=np.float32)
for i in tqdm(range (0,len(filesnames))):
    path=landmarks_folder+filesnames[i]+".npy"
    sample=np.load(path) #[138,468,2] 
    lens=set()  
    lens.add(len(sample)) 
    for j in range(len(sample)): #[468,2]
        frame=sample[j]
        sample[j]= standardize(frame)
        
    velocity=calc_velocity(sample)
    data=np.concatenate((sample[:-1,:,:], velocity), axis=2)
    lens=set()  
    lens.add(data.shape)
    normalized_data[i][:data.shape[0]]= data


#%%
normalized_data[7000]
#%%
path="/home/falhamdoosh/tgcn/data/PartA/Mediapipe/"
np.save(path+"dataset_mediapipe.npy",normalized_data)
#np.save(path+"label_mediapipe.npy",labels)
#%%

def get_mini_dataset(path):
    lis = []
    for _ in range(10):
        lis.append(np.random.randint(0, 51))
    idx_train=np.random.randint(low=0,high=8600,size=20)
    idx_test=np.random.randint(low= 0,high= 8600,size=20)
    print(idx_train,idx_test)
    
    np.save(path+"idx_train.npy",idx_train)
    np.save(path+"idx_test.npy",idx_test)

path="/home/falhamdoosh/tgcn/data/PartA/minidata/"
get_mini_dataset(path)

#%%
get_idx_train_test(path)

#%%
for i in tqdm(range (0,len(filesnames))):
    extract_landmarks_from_video(video_folder,filesnames[i])

#%%
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

for i in tqdm(range (0,len(filesnames))):
    extract_landmarks_from_video_media_pipe(video_folder,filesnames[i],mp_face_mesh)
