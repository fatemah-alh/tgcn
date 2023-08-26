#%%
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import re
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from utiles import get_file_names
from collections import Counter
(137,68,3),(137,51,3)
def extract_landmarks_video(feateure_extractor,video_path,output_path):  
    """
    Function to extract 3D landmarks from a single video using OpenFace library
    feateure_extractor: path to the feature extractor of OpenFace, 
    You shuold already have been installed the library and all dependencies.
    check the docomentation of OpenFace to see other flags and options.
    """ 
    command=f"{feateure_extractor} -f {video_path} -out_dir {output_path} -3Dfp"
    os.system(command)

def extract_all(feature_extrator,video_folder_path,landmarks_path,filesnames):
    """
    Function to iterate over all video and extract 3D landmarks.
    """
    for i in tqdm(range (0,len(filesnames))):#len(filesnames))
        path=video_folder_path+filesnames[i]+".mp4"
        out_path=landmarks_path+filesnames[i]
        extract_landmarks_video(feature_extrator,path,out_path)

def get_output_data(filesnames,landmarks_path):
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
    for i in tqdm(range (0,len(filesnames))):
        p_=landmarks_path+filesnames[i]+"/"+filesnames[i].split("/")[1]
        path=p_+".csv"
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
            os.system("rm "+path)
            os.system("rm "+p_+"_of_details.txt")
    dict_sorted=sorted(dict_count.items(), key=lambda x:x[1],reverse=True)
    idx_video_low_confidence=[i[0] for i in dict_sorted]
    #save idx to filter later
    #np.save("/home/falhamdoosh/tgcn/data/PartA/idx_low_confidance_90.npy",idx_video_low_confidence)
    return dict_count

        
if __name__=="__main__":
    """
    Before Running this file, check the config file and choos the name_file 
    
    """
    path_vis="/home/falhamdoosh/tgcn/data/PartA/vis/" # path to save gif of visualizzation
    name_file = 'open_face' # !IMPORTANT: name of conficuration file.
    config_file=open("./config/"+name_file+".yml", 'r')
    config = yaml.safe_load(config_file)
   
    video_folder_path=config["video_path"]
    landmarks_path=config["landmarks_path"]
    csv_file=config["csv_file"]
    feature_extrator="/Users/fatemahalhamdoosh/Desktop/Git/tesi/OpenFace/build/bin/FeatureExtraction" #TODO add your path to the feature extractor
    filesnames=get_file_names(csv_file)
    extract_all(feature_extrator,video_folder_path,landmarks_path,filesnames)
    dict=get_output_data(filesnames,landmarks_path)



