import numpy as np
import pandas as pd
import yaml
import sys
import os
import re
from tqdm import tqdm 
import mediapipe as mp
import cv2
parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)

from process_data.process_landmarks import *

class DataGenerator():
    def __init__(self,config,feature_extrator="/Users/fatemahalhamdoosh/Desktop/Git/PAIN_ESTIMATION/OpenFace/build/bin/FeatureExtraction"):
        self.config=config
        self.parent_folder=config["parent_folder"]
        self.video_folder_path=self.parent_folder+config["video_path"]
        self.landmarks_path=self.parent_folder+config["landmarks_path"]
        self.landmarks_npy=self.parent_folder+config["landmarks_npy"]
        self.csv_file=self.parent_folder+config["csv_file"]
        self.folder_data_path=self.parent_folder+config["folder_data_path"]
        self.feature_extrator=feature_extrator
        self.data_path=self.parent_folder+config["data_path"]
        self.labels_path=self.parent_folder+config["labels_path"]
        self.edges_path=self.parent_folder+config["edges_path"]
        self.idx_train_path=self.parent_folder+config["idx_train"]
        self.idx_test_path=self.parent_folder+config["idx_test"]
        self.filter_idx_90=parent_folder+config["filter_idx_90"]
        self.filesnames=get_file_names(self.csv_file)
        self.TS=config['TS']
        self.num_nodes=config['n_joints']
        self.dataset=config["dataset"]
        print(len(self.filesnames))

    def extract_all(self):
        """
        Function to iterate over all video and extract 3D landmarks.
        """
        for i in tqdm(range (0,len(self.filesnames))):#len(filesnames))
            path=self.video_folder_path+self.filesnames[i]+".mp4"
            out_path=self.landmarks_path+self.filesnames[i]
            command=f"{self.feature_extrator} -f {path} -out_dir {out_path} -3Dfp"
            os.system(command)
            
    def mediapipe(self):
        
        detector =  mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,min_tracking_confidence=0.5)

        for i in tqdm(range (0,len(self.filesnames))):
            self.mediapipe_video(self.filesnames[i],detector)
            

    def mediapipe_video(self,path_file,detector):
        path=self.video_folder_path+ path_file+".mp4"
        
        outputfile=self.landmarks_npy+path_file+".npy"
        
        os.makedirs(os.path.dirname(outputfile), exist_ok=True)
        cap = cv2.VideoCapture(path)
        list_landmarks=[]
        while True:
        # Read a frame from the video capture
            ret, frame = cap.read()

            if not ret:
                break
            data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks_=detector.process(data)
        
            if landmarks_.multi_face_landmarks:
                for face_landmarks in landmarks_.multi_face_landmarks:
                    landmarks = [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]
                   # landmarks_tensor = torch.tensor(landmarks)
                    #print(len(landmarks),len(landmarks[0]))
                    list_landmarks.append(landmarks)
                     # Or handle cases where no landmarks are detected
        cap.release()
        np.save(outputfile,list_landmarks)
        print(len(list_landmarks))
        return list_landmarks


    def get_output_data(self):
        """
        Function to process the csv file released by OpenFace and extract a matrix of landmarks position
        (137,68,3) 
        137: is number of frames 
        68: number of landmarks
        3: x,y,z coordinates
        This will save the npy matrix in each video folder and will delete the csv file and txt file 
        dict_count: is a dictionary whose keys are names of videos which contains frames with low confidence,
        and whose values are number of low confidence frames fuonds.
        
        """
        dict_count={}
        zero_frame=np.zeros((68,3))
        for i in tqdm(range (0,len(self.filesnames))):
            a=self.filesnames[i].split("/")
            p_=self.landmarks_npy+self.filesnames[i]+"/"+a[1]
            os.makedirs(self.landmarks_npy+self.filesnames[i]+"/",exist_ok=True)
            path=self.landmarks_path+self.filesnames[i]+"/"+self.filesnames[i].split("/")[1]+".csv"
            if(os.path.exists(path)):
                df = pd.read_csv(path)
                x_regex_pat = re.compile(r'^X_[0-9]+$')
                y_regex_pat = re.compile(r'^Y_[0-9]+$')
                z_regex_pat = re.compile(r'^Z_[0-9]+$')
                x_locs = df.columns[df.columns.str.contains(x_regex_pat)]
                y_locs = df.columns[df.columns.str.contains(y_regex_pat)]
                z_locs = df.columns[df.columns.str.contains(z_regex_pat)]
                sample=np.zeros((138,68,3))
                sample[:,:,0]=df[x_locs].to_numpy()
                sample[:,:,1]=df[y_locs].to_numpy()
                sample[:,:,2]=df[z_locs].to_numpy()
                #check the condidence of the frame.
                confidence=df["confidence"].to_numpy()
                success=df["success"].to_numpy()
                count=0
                for s in range(len(confidence)):
                    if success[s]!=1 or confidence[s]<0.9:
                        count=count+1
                        sample[s,:,:]=zero_frame # or insert the previous example sample[s-1,:,:]
                if(count!=0):
                    dict_count[i]=count

                np.save(p_,sample)
                #os.system("rm "+path)
                #os.system("rm "+p_+"_of_details.txt")
        dict_sorted=sorted(dict_count.items(), key=lambda x:x[1],reverse=True)
        idx_video_low_confidence=[i[0] for i in dict_sorted]
        #save idx to filter later
        np.save(self.folder_data_path+"idx_low_confidance_90.npy",idx_video_low_confidence)
        return dict_count
    def generate_landmarks(self):
        self.extract_all()
        dict=self.get_output_data()
    def generate_processed_data_set(self):
        normalized_data = np.zeros((len(self.filesnames), 68, self.num_nodes, 6), dtype=np.float32)
        process_all_data_new(landmarks_folder=self.landmarks_npy,
                             filesnames=self.filesnames,
                             normalized_data=normalized_data,
                             path_save=self.data_path,
                             down_sample=False)

    def generate_labels(self):
        save_labels(self.csv_file,self.labels_path)
        
    def generate_training_files_hold_out_filtered(self):
        split_idx_train_test(idx_train_path=self.idx_train_path,
                             idx_test_path=self.idx_test_path,
                             csv_file=self.csv_file,
                             filter_90=self.filter_idx_90)
        
    def generate_training_files_loso_ME67(self):
        split_loso_filter_ME(csv_file=self.csv_file,
                                filter_idx_90=self.filter_idx_90,
                                path=self.folder_data_path,
                                dataset=self.dataset,
                                num_subjects=67)
       
    def generate_training_files_loso_LE87(self):
        split_loso_filter_LE(filter_idx_90=self.filter_idx_90,
                                path=self.folder_data_path,
                                dataset=self.dataset,
                                num_subjects=86)
    
    def generate_cross_datasetAB(self):
        pass

    def generate_training_files_loso_ME67_crossDatasetAB(self):
        pass
       
    def generate_training_files_loso_LE87_crossDatasetAB(self):
        pass

    def generate_edges(self,landmarks=None):
        if landmarks==None:
            dataset_=np.load(self.data_path)
            landmarks=dataset_[0,0,:,:]
        get_edges(landmarks,edges_path=self.edges_path)
        
if __name__=="__main__":
    name_file = 'open_face_downsample' # !IMPORTANT: name of configuration file.
    config_file=open("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/config/"+name_file+".yml", 'r')
    config = yaml.safe_load(config_file)
    data_generator=DataGenerator(config)
    #data_generator.generat_landmarks()
    #data_generator.get_output_data()
    data_generator.generate_processed_data_set()
    #data_generator.generate_labels()
    #data_generator.generate_edges()
    #data_generator.generate_training_files_hold_out_filtered()
    #data_generator.generate_training_files_loso_ME67()
    #data_generator.generate_training_files_loso_LE87()
    #data_generator.extract_all_mediapipe()
    #data_generator.mediapipe()
   

