
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of utility functions for facial landmark processing 

@author: Vasileios Vonikakis
"""

import math
import random

import matplotlib.pyplot as plt
import cv2
#from mlxtend.image import extract_face_landmarks
import numpy as np

import pandas as pd
from tqdm import tqdm 
import os
import mediapipe as mp
import math

from numpy.linalg import eig

def get_file_names(csv_file):
    """
    Function to get a list of paths to videos to process. 
    """
    df = pd.read_csv(csv_file,sep='\t')
    filesnames=(df['subject_name'] + '/' + df['sample_name']).to_numpy()
    return filesnames

"""

def extract_dlib(filesnames,video_folder):
    for i in tqdm(range (0,len(filesnames))):
        extract_landmarks_from_video(video_folder,filesnames[i])
"""
def extract_mediaPipe(filesnames,video_folder):
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    for i in tqdm(range (0,len(filesnames))):
        extract_landmarks_from_video_media_pipe(video_folder,filesnames[i],mp_face_mesh)

def frontalize_landmarks_dlib(root,path_file,detector,predictor,frontalization_weights):
    path=root+"video/"+ path_file+".mp4"
    outputfile=root+"landmarks_frontalize/"+path_file+".npy"
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    cap = cv2.VideoCapture(path)
    list_landmarks=[]
    while True:
        # Read a frame from the video
        ret, frame = cap.read() 
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detector(image)
        if(len(face)>0):
            landmarks_raw = predictor(image,face[0])
            landmarks = get_landmark_array(landmarks_raw)
            
            landmarks_frontal = frontalize_landmarks(landmarks,frontalization_weights)
            
            list_landmarks.append(landmarks_frontal)
            
    cap.release()
    np.save(outputfile,list_landmarks)
    return list_landmarks
      

"""

def extract_landmarks_from_video(root,path_file,centroid=True):
    
   # This function take in input:
  #  root: Director where all videos are saved
   # path_file: a pathe insid videos folder to a video.mp4
   # Return:
   # list_landmarks:[num_frames,68,2]
   
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
"""
def extract_landmarks_from_video_media_pipe(root,path_file,mp_face_mesh,centroid=True):
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

def get_centroid(points):
    """
    Calulate the centroid of list of points(x,y)
    """
    #coords=torch.tensor(points).t().contiguous().squeeze()
    
    centroid =np.mean(points, axis=0)
    return centroid
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


def min_max_normalization(sample):
    face_min_max_x = (sample[:,0] - np.min(sample[:,0])) / (np.max(sample[:,0]) - np.min(sample[:,0]))
    face_min_max_y = (sample[:,1] - np.min(sample[:,1])) / (np.max(sample[:,1]) - np.min(sample[:,1]))
    return np.stack((face_min_max_x, face_min_max_y), axis=1)
def remove_rotation(sample):
    right_eye = sample[263]
    left_eye = sample[33]
    angle = angle_between(-left_eye + right_eye,[1,0] )
    matrix_x = rotation_matrix_2d(angle)
    rotated_points = np.dot(sample, matrix_x)
    return rotated_points.transpose()
def noise_centring(sample):
    return sample - sample[30]    


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


def get_mini_dataset(path="/home/falhamdoosh/tgcn/data/PartA/minidata/"):

    idx_train=np.random.randint(low=0,high=8600,size=20)
    idx_test=np.random.randint(low= 0,high= 8600,size=20)
    print(idx_train,idx_test)
    
    np.save(path+"idx_train.npy",idx_train)
    np.save(path+"idx_test.npy",idx_test)


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


def get_angle_degree(v1,v2):
    dY = v1[1] - v2[1]
    dX = v1[0] - v2[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    return angle

def align_points(arr):
    frontup=arr[263]#arr[10]
    frontdown=arr[33]#arr[152]
    angle = angle_between(frontup - frontdown ,[1,0] )
    matrix_x = rotation_matrix_2d(angle)

    rotated_points = np.dot(arr, matrix_x)
    rotated_points=np.dot(rotated_points,[[0,-1],[1,0]]) # 90
    return rotated_points


def align_points_degree(arr):
    rotated_points=arr
    leftEye=rotated_points[263]#arr[10]
    rightEye=rotated_points[33]#arr[152]
    
    eyesCenter = (np.mean([leftEye[0],rightEye[0]]),np.mean([leftEye[1],rightEye[1]]))
    #print(leftEye,rightEye,eyesCenter)
    angle =get_angle_degree(leftEye ,rightEye)
    M=cv2.getRotationMatrix2D(eyesCenter,angle,1)
    rotated_points=np.dot(rotated_points,M.T)
    return  rotated_points

def align_points_2(arr):
    rotated_points=arr
    #rotate intorn vertical axis
    
    point1=rotated_points[152] #frontal up
    point2=rotated_points[10]
    angle1 = angle_between(point2 -point1 ,[1,0] )
    matrix_x = rotation_matrix_2d(angle1)
    
    rotated_points = np.dot(rotated_points, matrix_x.T)

    eyeL=rotated_points[263]# #eyes
    eyeR=rotated_points[33]#eyes
    angle2 = angle_between(eyeL -eyeR ,[0,1] )
    matrix_x = rotation_matrix_2d(angle2)
    rotated_points = np.dot(rotated_points, matrix_x.T)
    #rotated_points=np.dot(rotated_points,[[0,1],[-1,0]]) # 270 
    #rotated_points=np.dot(rotated_points,[[0,-1],[1,0]]) # 90
    #rotated_points=np.dot(rotated_points,[[0,1],[-1,0]]) # 270 
    return rotated_points





def split_data_test_train_balance(csv_file,data,labels):

    """
    ####Not used####
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
    df = pd.read_csv(csv_file,sep='\t')
    mask = df['subject_name'].isin(validation_subjects_id)
    idx_test= df.loc[mask].index.tolist()
    test_data=data[idx_test]
    test_labels=labels[idx_test]
    train_data=data[~idx_test]
    train_labels=labels[~idx_test]
    assert len(train_data)==len(train_labels)
    assert len(test_data)==len(test_labels)
    return (test_data,test_labels,train_data,train_labels)

           
def rotate_z(landmarks):
    angles = np.array([45, 90, 60])
    angles_rad = np.radians(angles)
    rotation_z = np.array([[np.cos(angles_rad[2]), -np.sin(angles_rad[2]), 0],
                           [np.sin(angles_rad[2]), np.cos(angles_rad[2]), 0],
                           [0, 0, 1]])

    # Apply sequential rotations
    rotated_points = np.dot(rotation_z, landmarks.T).T
    return rotated_points
    

def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]  # pingyi bianhuan
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy
def get_eigenvectors(arr,ord=True):
    cova=np.cov(arr,rowvar=False)
    eigenvalues, eigenvectors = eig(cova)
    if ord:
        idx = (eigenvalues).argsort()[::-1]   
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
    return eigenvectors
def align_eigenvectors(eigenvectors,arr):
    arr=np.dot(eigenvectors, arr.T).T # 68*3 __ 3*3 

    return arr


def get_landmark_array(landmarks_dlib_obj):
    # Gets a DLIB landmarks object and returns a [68,2] numpy array with the 
    # landmark coordinates.
    
    landmark_array = np.zeros([68,2])
    
    for i in range(68):
        landmark_array[i,0] = landmarks_dlib_obj.part(i).x
        landmark_array[i,1] = landmarks_dlib_obj.part(i).y
        
    return landmark_array  # numpy array



def frontalize_landmarks(landmarks, frontalization_weights):
    # 
    
    '''
    ---------------------------------------------------------------------------
                      Frontalize a non-frontal face shape
    ---------------------------------------------------------------------------
    Takes an array or a list of facial landmark coordinates and returns a 
    frontalized version of them (how the face shape would look like from 
    the frontal view). Assumes 68 points with a DLIB annotation scheme. As
    described in the paper: 
    V. Vonikakis, S. Winkler. (2020). Identity Invariant Facial Landmark 
    Frontalization for Facial Expression Analysis. ICIP2020, October 2020.
    
    INPUTS
    ------
    landmarks: numpy array [68,2]
        The landmark array of the input face shape. Should follow the DLIB 
        annotation scheme.
    frontalization_weights: numpy array of [68x2+1, 68x2]
        The frontalization weights, learnt from the fill_matrices.py script. 
        The +1 is due to the interception during training. 

    OUTPUT
    ------
    landmarks_frontal: numpy array [68,2]
        The landmark array of the frontalized input face shape. 
    '''
    
    if type(landmarks) is list:
        landmarks = get_landmark_matrix(landmarks)
    
    landmarks_standard = get_procrustes(landmarks, template_landmarks=None)
    landmark_vector = np.hstack(
        (landmarks_standard[:,0].T, landmarks_standard[:,1].T, 1)
        )  # add interception
    landmarks_frontal = np.matmul(landmark_vector, frontalization_weights)
    landmarks_frontal = get_landmark_matrix(landmarks_frontal)
    
    return landmarks_frontal



def get_landmark_matrix(ls_coord):
    # Gets a list of landmark coordinates and returns a [N,2] numpy array of
    # the coordinates. Assumes that the list follows the scheme
    # [x1, x2, ..., xN, y1, y2, ..., yN]
    
    mid = len(ls_coord) // 2
    landmarks = np.array( [ ls_coord[:mid], ls_coord[mid:] ]    )
    return landmarks.T

def my_random_generator(n_values, min_value, max_value, excluded_values=None):
    """
    fuction to generate list of random integers excluding values in list "excluded_values"
    usage :
    l = list(my_random_generator(100, 0, 8650, idx_train_data))
    """
    if excluded_values is None:
        excluded_values = []

    count = 0

    while count < n_values:
        value = np.random.randint(min_value, max_value)
        while value in excluded_values:
            value = np.random.randint(min_value, max_value)
        yield value
        count += 1
        


def rotate_x(landmarks,v="none"):
    if v=="none":
        right_eye = landmarks[36]
        left_eye = landmarks[45]
        v=left_eye - right_eye
    
    angle= angle_between( v,[1,0,0] )
    
    
    rotation_x = np.array([[1, 0, 0],
                           [0, math.cos(angle), -math.sin(angle)],
                           [0, math.sin(angle), math.cos(angle)]])
    rotated_points = np.dot(rotation_x, landmarks.T).T

    return rotated_points

def rotate_y(landmarks,v="none"):
    if v=="none":
        right_eye = landmarks[36]
        left_eye = landmarks[45]
        v=left_eye - right_eye
    angle= angle_between(v,[0,1,0] )
   
   # angles = np.array([60,20, 60])
   # angles= np.radians(angles)
    rotation_y = np.array([[math.cos(angle), 0, math.sin(angle)],
                           [0, 1, 0],
                           [-math.sin(angle), 0, math.cos(angle)]])

    rotated_points = np.dot(rotation_y, landmarks.T).T
    return rotated_points

def rotate_z(landmarks, v="none"):
    if v=="none":
        right_eye = landmarks[36]
        left_eye = landmarks[45]
        v=left_eye - right_eye
        print("none")
    
    angle= angle_between(v,[0,0,1] )
    
    rotation_z = np.array([[math.cos(angle), -math.sin(angle), 0],
                           [math.sin(angle), math.cos(angle), 0],
                           [0, 0, 1]])

    # Apply sequential rotations
    rotated_points = np.dot(rotation_z, landmarks.T).T
    return rotated_points



def mirror_landmarks(landmarks):
    # Given a numpy array of [N,2] facial landmarks, it returns the mirrored
    # face shape. It assumes notmalized lanmark coordinates where the mean of 
    # the face shape is 0.
    
    landmarks_inverted = landmarks.copy()
    landmarks_inverted[:,0] = - landmarks_inverted[:,0]  # x = - x
    
    return landmarks_inverted



def get_eye_centers(landmarks):
    # Given a numpy array of [68,2] facial landmarks, returns the eye centers 
    # of a face. Assumes the DLIB landmark scheme.

    landmarks_eye_left = landmarks[36:42,:]
    landmarks_eye_right = landmarks[42:48,:]
    
    center_eye_left = np.mean(landmarks_eye_left, axis=0)
    center_eye_right = np.mean(landmarks_eye_right, axis=0)
    
    return center_eye_left, center_eye_right



def get_procrustes(
        landmarks, 
        translate=True, 
        scale=True, 
        rotate=True, 
        template_landmarks=None):
    '''
    ---------------------------------------------------------------------------
                        Procrustes shape standardization
    ---------------------------------------------------------------------------
    Standardizes a given face shape, compensating for translation, scaling and
    rotation. If a template face is also given, then the standardized face is
    adjusted so as its facial parts will be displaced according to the 
    template face. More information can be found in this paper:
        
    V. Vonikakis, S. Winkler. (2020). Identity Invariant Facial Landmark 
    Frontalization for Facial Expression Analysis. ICIP2020, October 2020.
    
    INPUTS
    ------
    landmarks: numpy array [68,2]
        The landmark array of the input face shape. Should follow the DLIB 
        annotation scheme.
    translate: Boolean
        Whether or not to compensate for translation.
    scale: Boolean
        Whether or not to compensate for scaling.
    rotation: Boolean
        Whether or not to compensate for rotation.
    template_landmarks: numpy array [68,2] or None
        The landmark array of a template face shape, which will serve as 
        guidence to displace facial parts. Should follow the DLIB 
        annotation scheme. If None, no displacement is applied. 
    
    OUTPUT
    ------
    landmarks_standard: numpy array [68,2]
        The standardised landmark array of the input face shape.
        
    '''
    
    landmarks_standard = landmarks.copy()
    
    # translation
    if translate is True:
        landmark_mean = np.mean(landmarks, axis=0)
        landmarks_standard = landmarks_standard - landmark_mean
    
    # scale
    if scale is True:
        landmark_scale = math.sqrt(
            np.mean(np.sum(landmarks_standard**2, axis=1))
            )
        landmarks_standard = landmarks_standard / landmark_scale
    
    
    if rotate is True:
        # rotation
        center_eye_left, center_eye_right = get_eye_centers(landmarks_standard)
        
        # distance between the eyes
        dx = center_eye_right[0] - center_eye_left[0]
        dy = center_eye_right[1] - center_eye_left[1]
    
        if dx != 0:
            f = dy / dx
            a = math.atan(f)  # rotation angle in radians
            # ad = math.degrees(a)
            # print('Eye2eye angle=', ad)
    
        R = np.array([
            [math.cos(a), -math.sin(a)], 
            [math.sin(a), math.cos(a)]
            ])  # rotation matrix
        landmarks_standard = np.matmul(landmarks_standard, R)
    
    '''
    adjusting facial parts to a tamplate face
    displacing face parts to predetermined positions (as defined by the 
    template_landmarks), except from the eyebrows, which convey important 
    expression information attention! this only makes sense for frontal faces!
    '''
    if template_landmarks is not None:
        
        # mouth
        anchorpoint_template = np.mean(template_landmarks[50:53,:], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[50:53,:], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[48:,:] += displacement
        
        # right eye
        anchorpoint_template = np.mean(template_landmarks[42:48,:], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[42:48,:], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[42:48,:] += displacement
        # right eyebrow (same displaycement as the right eye)
        landmarks_standard[22:27,:] += displacement  # TODO: only X?
        
        # left eye
        anchorpoint_template = np.mean(template_landmarks[36:42,:], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[36:42,:], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[36:42,:] += displacement
        # left eyebrow (same displaycement as the left eye)
        landmarks_standard[17:22,:] += displacement  # TODO: only X?
        
        # nose
        anchorpoint_template = np.mean(template_landmarks[27:36,:], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[27:36,:], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[27:36,:] += displacement
        
        # jaw
        anchorpoint_template = np.mean(template_landmarks[:17,:], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[:17,:], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[:17,:] += displacement
        
    return landmarks_standard


