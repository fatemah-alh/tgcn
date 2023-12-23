#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import sys
import os
parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)
from main import Trainer
from helper.Config import Config
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import random


def get_all_outputs(config,path_pretrained_model):
    trainer=Trainer(config=config)
    return trainer.get_all_outputs(path=path_pretrained_model)
def get_video_paths(csv_file,idx_path,videos_folder):
    idx=np.load(idx_path)
    df=pd.read_csv(csv_file,sep='\t')
    r=df.iloc[idx]
    r['path']=videos_folder+r['subject_name']+"/"+r['sample_name']+".mp4"
    return r['path'].tolist()
def get_id_subjects(csv_file,path):
    df=pd.read_csv(csv_file,sep='\t')
    result = df.groupby(['subject_id', 'subject_name']).size().reset_index().rename(columns={0: 'count'})
    result=result[['subject_id', 'subject_name']]
    result.to_csv(path)
def video_to_images(video_path, target_length=138, fill_color='black'):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    images = []
    while True:
        # Read each frame
        ret, frame = cap.read()

        # Break the loop if there are no frames left
        if not ret:
            break

        # Append the frame (image) to the list
        images.append(frame)

    # Release the video capture object
    cap.release()

    # Fill in remaining frames if necessary
    current_length = len(images)
    if current_length < target_length:
        # Determine the color for filling (black or white)
        fill_value = 0 if fill_color == 'black' else 255
        
        # Size of the images (assuming all frames have the same size)
        height, width, channels = images[0].shape if images else (480, 640, 3)

        # Add additional frames
        for _ in range(target_length - current_length):
            filler_frame = np.full((height, width, channels), fill_value, dtype=np.uint8)
            images.append(filler_frame)

    return images

def plot_with_img(images, line_data,landmarks, title, output_file='output.mp4', fps=30):
    """
    Save a video from given frames, line plots, and texts.

    Args:
    images (list): List of image file paths.
    line_data (list): List of tuples (x, y) for line plot data.
    texts (list): List of texts for each frame.
    output_file (str): Output file name.
    fps (int): Frames per second for the video.
    """
    #Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #cv2.VideoWriter_fourcc('m', 'p', '4', 'v')#
    height, width,channel=images[0].shape
    height, width=height//2, width//2
    dpi=100
    width_in = width / dpi
    height_in = height / dpi
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    video = cv2.VideoWriter(output_file, fourcc, fps, (width*2, height*2),True)

    for i, (img, x) in enumerate(zip(images, line_data)):
        # Create an inset with size and position (x, y, width, height) in figure coordinates
        if i>6 and i<130:
            # Create plot as image
            plt.figure(figsize=(width_in*2, height_in), dpi=dpi)
            plt.plot(line_data[7:130])
            plt.plot(i-7, x, 'o')
            plt.title(title+f", Current VAS: {x}")
            plt.xlabel('Frame Index')
            plt.ylabel('VAS Pain Score')
            plt.savefig('temp_plot.png', dpi=dpi)
            plt.close()
            plt.figure(figsize=(width_in, height_in), dpi=dpi)
            fram_fl=landmarks[i]
            
            plt.scatter(fram_fl[:,0], fram_fl[:,1],alpha=0.7, c="blue")
            plt.savefig('fl.png', dpi=dpi)
            plt.close()
            fl_img=cv2.imread('fl.png')
            plot_img = cv2.imread('temp_plot.png')
            # Combine image and plot
            img = cv2.resize(img, (width, height))
            comb_1=cv2.hconcat([img,fl_img])
            cv2.imwrite('comb_1.jpg', comb_1)
            comb_1=cv2.imread('comb_1.jpg')
            combined_img = cv2.vconcat([comb_1, plot_img])
            cv2.imwrite('combined_image.jpg', combined_img)
            combined_img = cv2.imread('combined_image.jpg')
            video.write(combined_img)

    # Release the video writer
    video.release()

def plot_videos(val_target,val_predic,targets,predicted,outputs,features,videos_paths,output_folder,num_plot=10,random_=True):
    idx_selected=torch.where((targets==val_target) & (predicted==val_predic))[0]
    videos_paths_selected=[videos_paths[id] for id in idx_selected]
    outputs_selected=outputs[idx_selected]
    targets_selected=targets[idx_selected]
    predicted_selected=predicted[idx_selected]
    num_= num_plot if len(idx_selected)> num_plot else len(idx_selected)
    landmarks= reshape_data(features[idx_selected])
    if num_>0:
        idx_random=random.sample(range(0, len(idx_selected)), num_)
        for v in idx_random : #range(0,len(videos_paths_selected)):
            video_name=os.path.basename(os.path.normpath(videos_paths_selected[v]))
            images =video_to_images(videos_paths_selected[v]) 
            plot_with_img(images=images[:-1],
                        line_data=outputs_selected[v],
                        landmarks=landmarks[v],
                        title=f"Target:{targets_selected[v]},Predicted:{predicted_selected[v]}\n",
                        output_file=output_folder+f"t_{val_target}_p_{val_predic}/{video_name}")
def reshape_data(features):
    reshaped_tensor = np.transpose(features, (0, 2, 3, 1,4))  # Transpose dimensions from 8700,137,469,4) to 8600, 469, 4, 137
    new_feat= np.reshape(reshaped_tensor, (len(features), 137,51,6,1)).squeeze(axis=-1) 
    return new_feat[:,:,:,:2]
def main(labels_t=[0,1,2,3,4],labels_p=[0,1,2,3,4],num_plot=2,folder="gru_outputs/",random_=True):
    config_file="open_face"
    parent_folder="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
    config =Config.load_from_file(parent_folder+"/config/"+config_file+".yml")
    videos_folder="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/video/"
    output_folder="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/vis/"+folder
    model_path="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/Old/1s+15k+multi+MinMaxNorm_xy+NoLabelNorm/best_model.pkl"
    videos_paths=get_video_paths(config.parent_folder+config.csv_file,config.parent_folder+config.idx_test,videos_folder)
    targets,predicted,outputs,features=get_all_outputs(config,model_path)
    #labels=[0,1,2,3,4]
    for l in labels_t:
        for ll in labels_p:
            plot_videos(l,ll,targets,predicted,outputs,features,videos_paths,output_folder,num_plot=num_plot,random_=random_)

#%%

main(labels_t=[4],labels_p=[4],num_plot=40,folder="t4/",random_=False)
# %%

# %%
