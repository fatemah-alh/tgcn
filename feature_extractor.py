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


name_exp = 'biovid'
config_file=open("./config/"+name_exp+".yml", 'r')
config = yaml.safe_load(config_file)

#%%
if torch.cuda.is_available():
  torch.cuda.set_device(0)
  device = 'cuda'  
else:
  device='cpu'

dlib.cuda.set_device(0)

def get_centroid(points):
    """
    Calulate the centroid of list of points(x,y)
    """
    #coords=torch.tensor(points).t().contiguous().squeeze()
    
    centroid =np.mean(points, axis=0)
    return centroid
def extract_landmarks_from_video(root,path_file,centroid=True):
    """
    This function take in input:
    root: Director where all videos are saved
    path_file: a pathe insid videos folder to a video.mp4
    Return:
    list_landmarks:[num_frames,68,2]
    """
    #path='/home/falhamdoosh/tgcn/data/PartA/video/071309_w_21/071309_w_21-BL1-081.mp4'
    path=root+"video/"+ path_file+".mp4"
    outputfile=root+"landmarks/"+path_file+".npy"
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    cap = cv2.VideoCapture(path)
    list_landmarks=[]
    while True:
        # Read a frame from the video
        ret, frame = cap.read() 
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract the landmarks of the face
        #data=torch.from_numpy(gray).to(device)
        landmarks = extract_face_landmarks(gray)
        
        if centroid:
            landmarks=np.vstack((landmarks,get_centroid(landmarks)))
        list_landmarks.append(landmarks)

    # Release the resources
    cap.release()
    np.save(outputfile,list_landmarks)
    return list_landmarks

def extract_landmarks_from_video_media_pipe(root,path_file,centroid=True):
    """
    This function take in input:
    root: Director where all videos are saved
    path_file: a pathe insid videos folder to a video.mp4
    Return:
    list_landmarks:[num_frames,68,2]
    """
    #path='/home/falhamdoosh/tgcn/data/PartA/video/071309_w_21/071309_w_21-BL1-081.mp4'
    path=root+"video/"+ path_file+".mp4"
    outputfile=root+"landmarks/"+path_file+".npy"
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    cap = cv2.VideoCapture(path)
    list_landmarks=[]
    while True:
    # Read a frame from the video capture
        ret, frame = cap.read()

        if not ret:
            break
        data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = mp_face_mesh.process(data)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert normalized landmarks to pixel coordinates
                h, w, _ = frame.shape
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))
                if centroid:
                    landmarks=np.vstack((landmarks,get_centroid(landmarks)))
                    list_landmarks.append(landmarks)
    cap.release()
    np.save(outputfile,list_landmarks)
    return list_landmarks
def filter_neutral_subject(df):

    neutral_subjects_id=["082315_w_60", "082414_m_64", "082909_m_47", "083009_w_42",
                        "083013_w_47", "083109_m_60", "083114_w_55", "091914_m_46",
                        "092009_m_54", "092014_m_56", "092509_w_51", "092714_m_64", 
                        "100514_w_51","100914_m_39", "101114_w_37","101209_w_61", 
                        "101809_m_59", "101916_m_40", "111313_m_64", "120614_w_61"]
    mask = df['subject_name'].isin(neutral_subjects_id)
    indices_filtered= df.loc[mask].index.tolist()
    df=df[~mask]
    return indices_filtered
#%%
def get_idx_train_test():
    validation_subjects_id=["100914_m_39", "101114_w_37", "082315_w_60", "083114_w_55", 
                            "083109_m_60", "072514_m_27", "080309_m_29", "112016_m_25", 
                            "112310_m_20", "092813_w_24", "112809_w_23", "112909_w_20", 
                            "071313_m_41", "101309_m_48", "101609_m_36", "091809_w_43", 
                            "102214_w_36", "102316_w_50", "112009_w_43", "101814_m_58", 
                            "101908_m_61", "102309_m_61", "112209_m_51", "112610_w_60", 
                            "112914_w_51", "120514_w_56"]
    labels_path=config['labels_path']
    label_file= open(labels_path,'rb')
    labels=pickle.load(label_file)
    print(labels[0])
    idx_test=[]
    idx_train=[]
    for i in range(len(labels[0])):
        test=False
        for j in validation_subjects_id:
            if labels[0][i].startswith(j):
                test=True
        if test:
            idx_test.append(i)        
        else:
            idx_train.append(i)

    print("Percent test",(len(idx_test)*100)/len(labels[0]))
    assert len(idx_test)+len(idx_train)==len(labels[0])
    return idx_train,idx_test
idx_train,idx_test=get_idx_train_test()

#%%
np.save("/home/falhamdoosh/tgcn/Painformer/idx_train.npy",idx_train)
np.save("/home/falhamdoosh/tgcn/Painformer/idx_test.npy",idx_test)
#%%
def split_test_train_balance(df,data,labels):

    """
    Split in train and test data according to Biovid indication to obtain balance.
    inputs:
            df: datafram of samples.csv
            data:matrix of all landmarks of all video [num of samples,[num_frames,68 or 69 if centroid was added,2]]
            labels: one dimension array of labels between 0 and 4 of size [ num of samples ] 
    Return:
            test_data,test_labels,train_data,train_labels
    """
    validation_subjects_id=["100914_m_39", "101114_w_37", "082315_w_60", "083114_w_55", 
                            "083109_m_60", "072514_m_27", "080309_m_29", "112016_m_25", 
                            "112310_m_20", "092813_w_24", "112809_w_23", "112909_w_20", 
                            "071313_m_41", "101309_m_48", "101609_m_36", "091809_w_43", 
                            "102214_w_36", "102316_w_50", "112009_w_43", "101814_m_58", 
                            "101908_m_61", "102309_m_61", "112209_m_51", "112610_w_60", 
                            "112914_w_51", "120514_w_56"]
    mask = df['subject_name'].isin(validation_subjects_id)
    indices_validation= df.loc[mask].index.tolist()
    #df_test=df[mask]
    #df_train=df[~mask]
    test_data=data[indices_validation]
    test_labels=labels[indices_validation]
    train_data=data[~indices_validation]
    train_labels=labels[~indices_validation]
    assert len(train_data)==len(train_labels)
    assert len(test_data)==len(test_labels)
    return (test_data,test_labels,train_data,train_labels)

def preprocess_data(data,standardization =True, minMaxNormalization=False):
    """
    This function perform data preproccessing:
        normalize and standarize points, calculate velocities of points.
    Inputs:
        data: matrix of all landmarks of all video [num of samples,[num_frames,68 or 69 if centroid was added,2]] 
        standardization: True if you want to standarize data points to zero mean and unitary std
        minMaxNormalization: True if you want to apply min max normalization
    """
    preprocess_data=[]
    for sample in range(len(data)):
        for frame in sample:
            pass

    return data

def min_max_normalization(sample):
    # Min-Max normalization
    face_min_max_x = (sample[:,0] - np.min(sample[:,0])) / (np.max(sample[:,0]) - np.min(sample[:,0]))
    face_min_max_y = (sample[:,1] - np.min(sample[:,1])) / (np.max(sample[:,1]) - np.min(sample[:,1]))
    return np.stack((face_min_max_x, face_min_max_y), axis=1)
def remove_rotation(sample):
    right_eye = sample[36]
    left_eye = sample[45]
    angle = angle_between(left_eye - right_eye, [1, 0])
    matrix_x = rotation_matrix_2d(angle)
    return np.dot(sample, matrix_x).transpose()
def noise_centring(sample):

    return sample - sample[30]
            
def calc_velocity(sample):
    velcetiy=[]
    for i in range(1,len(sample)):
        velcetiy.append(sample[i]-sample[i-1])
    return velcetiy  

def standardize(values):
    return (values - values.mean())/values.std()

def angle_between(v1, v2):
    '''
    Function that return the angle between 2 vectors.
    Angle is in radians.

    Parameters
    ----------
    v1 : Vector
        Coordinates of the vector v1
    v2 : Vector
        Coordinates of the vector v2

    Returns
    -------
    Float
        Angle between vectors v1 and v2 in radians

    '''
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_prod = np.dot(unit_v1, unit_v2)
    return np.arccos(np.clip(dot_prod, -1.0, 1.0))

def rotation_matrix_2d(theta):
    '''
    Function to get the 2D rotation matrix for a given angle theta (in radians).

    Parameters
    ----------
    theta : Float
        Angle in radians

    Returns
    -------
    Array
        The 2D rotation matrix for the given angle theta

    '''
    return np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])



video_folder="/home/falhamdoosh/tgcn/data/PartA/"
df = pd.read_csv('/home/falhamdoosh/tgcn/data/PartA/samples.csv',sep='\t')
labels=df['class_id'].values
filesnames=filesnames=(df['subject_name'] + '/' + df['sample_name']).to_numpy()

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
    extract_landmarks_from_video_media_pipe(video_folder,filesnames[i])
#%%
#TODO apply function to create data array
data=[]
np.save("/home/falhamdoosh/tgcn/data/PartA/data_landmarks.npy",data)
np.save("/home/falhamdoosh/tgcn/data/PartA/labels.npy",labels)

#%%
#Process_data
FILTER_NEUTRAL=False
FILTER_VALIDATE=True
#there is many subject in common within validation and neutral. so choose some of them.
