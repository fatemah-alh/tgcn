#%%
import numpy as np
import pandas as pd
from tqdm import tqdm 
import mediapipe as mp
from utiles import process_data,extract_landmarks_from_video_media_pipe,extract_landmarks_from_video

def extract_dlib(filesnames,video_folder):
    for i in tqdm(range (0,len(filesnames))):
        extract_landmarks_from_video(video_folder,filesnames[i])

def extract_mediaPipe(filesnames,video_folder):
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    for i in tqdm(range (0,len(filesnames))):
        extract_landmarks_from_video_media_pipe(video_folder,filesnames[i],mp_face_mesh)

def get_mini_dataset(path):
    path="/home/falhamdoosh/tgcn/data/PartA/minidata/"
    lis = []
    for _ in range(10):
        lis.append(np.random.randint(0, 51))
    idx_train=np.random.randint(low=0,high=8600,size=20)
    idx_test=np.random.randint(low= 0,high= 8600,size=20)
    print(idx_train,idx_test)
    
    np.save(path+"idx_train.npy",idx_train)
    np.save(path+"idx_test.npy",idx_test)

#%%

landmarks_folder_mediapipe="/home/falhamdoosh/tgcn/data/PartA/landmarks/"
video_folder="/home/falhamdoosh/tgcn/data/PartA/"
path_save="/home/falhamdoosh/tgcn/data/PartA/Mediapipe/"

df = pd.read_csv('/home/falhamdoosh/tgcn/data/PartA/samples.csv',sep='\t')

filesnames=(df['subject_name'] + '/' + df['sample_name']).to_numpy()

normalized_data = np.zeros((8700, 137, 469, 4), dtype=np.float32)

#%%
process_data(landmarks_folder_mediapipe,filesnames,normalized_data,path_save)


#save labels

#labels=df['class_id'].values
#np.save(path+"label_mediapipe.npy",labels)
#get_mini_dataset()
#get_idx_train_test(path)