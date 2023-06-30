#%%
import numpy as np
import pandas as pd
from tqdm import tqdm 
import mediapipe as mp


from utiles import frobenius_norm,align_points,extract_landmarks_from_video_media_pipe,calc_velocity,standardize,extract_landmarks_from_video

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


def process_data(landmarks_folder:str,filesnames:list,normalized_data:np.array,path_save:str):
    """
    Iterate over all landmarks.npy for each sample, apply
    1- make coordiates have zero mean and unitary std.
    2- remove rotaione of the face, align points to be in frontal face position. 
    3- Divise each frame by frobenius norm
    4- calculate velocitie
    create the data matrix that containes all samples [num_samples,num_frame,num_landmarks,num_featuers]
    """
    
    for i in tqdm(range (0,len(filesnames))):
        path=landmarks_folder+filesnames[i]+".npy"
        sample=np.load(path) #[138,468,2] 
        for j in range(len(sample)): #[468,2]
            frame=sample[j]
            frame= standardize(frame)
          #  frame=align_points(frame)
            sample[j]=frobenius_norm(frame)
        velocity=calc_velocity(sample)
        data=np.concatenate((sample[:-1,:,:], velocity), axis=2)
        normalized_data[i][:data.shape[0]]= data
    np.save(path_save+"dataset_mediapipe_without_process.npy",normalized_data)
    return normalized_data

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