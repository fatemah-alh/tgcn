import numpy as np
import pandas as pd
import yaml
import sys
import os
import re
from tqdm import tqdm 


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
        self.get_filenames()
    def get_filenames(self):
        """
        Function to get a list of paths to videos to process. 
        """
        df = pd.read_csv(self.csv_file,sep='\t')
        self.filesnames=(df['subject_name'] + '/' + df['sample_name']).to_numpy()
        return self.filesnames
    def extract_all(self):
        """
        Function to iterate over all video and extract 3D landmarks.
        """
        for i in tqdm(range (0,len(self.filesnames))):#len(filesnames))
            path=self.video_folder_path+self.filesnames[i]+".mp4"
            out_path=self.landmarks_path+self.filesnames[i]
            self.extract_landmarks_video(path,out_path)
    def extract_landmarks_video(self,video_path,output_path):  
        """
        Function to extract 3D landmarks from a single video using OpenFace library
        feateure_extractor: path to the feature extractor of OpenFace, 
        You shuold already have been installed the library and all dependencies.
        check the docomentation of OpenFace to see other flags and options.
        """ 
        command=f"{self.feature_extrator} -f {video_path} -out_dir {output_path} -3Dfp"
        os.system(command)
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
            p_=self.landmarks_npy+a[0]+"/"+a[1]
            os.makedirs(self.landmarks_npy+a[0]+"/",exist_ok=True)
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
    def generat_landmarks(self):
        self.extract_all()
        dict=self.get_output_data()
    def generat_processed_data_set(self):
        pass
    def generat_training_files(self):
        pass
if __name__=="__main__":
    name_file = 'open_face_PartB' # !IMPORTANT: name of conficuration file.
    config_file=open("/Users/fatemahalhamdoosh/Desktop/Git/PAIN_ESTIMATION/"+name_file+".yml", 'r')
    config = yaml.safe_load(config_file)
    data_generator=DataGenerator(config)
    #data_generator.generat_landmarks()
    #data_generator.get_output_data()
    idx=np.load("/Users/fatemahalhamdoosh/Desktop/Git/PAIN_ESTIMATION/data/PartB/idx_low_confidance_90.npy")
    print(len(idx),idx)

