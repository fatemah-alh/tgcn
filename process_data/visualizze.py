
#%%
import matplotlib.pyplot as plt
from sklearn.metrics import  ConfusionMatrixDisplay
import numpy as np
from tqdm import tqdm 
import yaml
from PIL import Image
import imageio
import sys
from scipy.spatial import Delaunay
import torch
from torch_geometric.utils import add_self_loops,to_undirected,to_dense_adj,contains_self_loops,is_undirected
import seaborn as sns
parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)

from helper.dataloader import DataLoader,Rotate,FlipV
from torchvision.transforms import RandomApply,RandomChoice,Compose
import wandb
from utiles import rotation_matrix_2d
import pandas as pd
def get_edges(landmarks,edges_path):
    tri = Delaunay(landmarks[:,:2])
    edges = []
    for simplex in tri.simplices:
        for i in range(3):
            edges.append((simplex[i], simplex[(i+1)%3]))
    edges_index=[[],[]]
    for e in edges:
        edges_index[0].append(e[0])
        edges_index[1].append(e[1])
    edges_index=torch.LongTensor(edges_index)
    
    print("number of index before adding symmetric edges:",edges_index.shape)
    print(len(edges_index[0]))
    #edges_index=to_undirected(edges_index)
    #edges_index_with_loops=add_self_loops(edges_index)
    #edges_index=edges_index_with_loops[0]
    print("Contian self loops:",contains_self_loops(edges_index))
    print("Graph is now undircte:",is_undirected(edges_index))
    print("number of index after adding symmetric edges:",edges_index.shape)
    np.save(edges_path,edges_index)
    return edges_index
    
def visualize_landmarks(data,label_data,edges=[],time_steps=137,vis_edges=False,vis="2d",vis_index=False,save=False,path_vis=None):
    """
    this function will visualize landmarks of data tensor, just the f
    """
    
    figures=[]
   
    for time_step in tqdm(range(time_steps)):
        # Create a figure and axes for subplots
        if vis=="2d":
            fig, axs = plt.subplots(5, 4, figsize=(30, 30))
        else:
            fig, axs = plt.subplots(5, 4, figsize=(30, 30),subplot_kw=dict(projection='3d'))
        
        #Flatten the axes array
        axs = axs.flatten()
        for d in range(0,20):
            fram=data[d,time_step,:,:]
            axs[d].set_title(f"VAS level: {label_data[d]}")
            if vis_index:
                for index in range(len(fram)):
                    axs[d].annotate(index,(fram[index][0],fram[index][1]))
            if vis=="2d":
                axs[d].scatter(fram[:,0], fram[:,1], alpha=0.8,s=1)
            else:
                axs[d].scatter(fram[:,0], fram[:,1],fram[:,2], alpha=0.8)
                #axs[d].view_init(-30,60)
            
            if vis_edges:
                for i in range(len(edges[0])):
                    axs.plot([fram[edges[0][i],0].item(), (fram[edges[1][i],0]).item()], [fram[edges[0][i],1].item(), (fram[edges[1][i],1]).item()], "blue",alpha=0.3)
            
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        figures.append(image)
    if save:
        imageio.mimsave(path_vis+'vis_landmarks.gif', figures, duration=50)

def visualize_sample(data,label_data,edges=[],time_steps=137,vis_edges=False,vis="2d",vis_index=False,save=False,path_vis=None):
    figures=[]
    for time_step in tqdm(range(time_steps)):
        # Create a figure and axes for subplots
        fig = plt.figure()
        if vis=="2d":
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(projection='3d')
        # Flatten the axes array
        fram=data[time_step,:,:]

        print(fram.shape)
        ax.set_title(f"VAS level: {label_data}")

        if vis_edges:
            for i in range(len(edges[0])):
                ax.plot([fram[edges[0][i],0].item(), (fram[edges[1][i],0]).item()], [fram[edges[0][i],1].item(), (fram[edges[1][i],1]).item()], "slateblue",alpha=0.2)
            
        if vis_index:
            for index in range(len(fram)):
                ax.annotate(index,(fram[index][0],fram[index][1]))
        if vis=="2d":
            ax.scatter(fram[:,0], fram[:,1],alpha=1, c="slateblue",s=50)
            
        else:
            ax.scatter(fram[:,0], fram[:,1],fram[:,2], c="slateblue")
            ax.view_init()
        
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        figures.append(image)
    if save:
        imageio.mimsave(path_vis+'vis_landmarks_3D_openface_absolut_eigenvectors_without_contur.gif', figures, duration=50)

def visualize_temporal_edges(data):
    fig = plt.figure()
   
    ax = fig.add_subplot()

    fram=data[0,:,:]
    fram2=data[1,:,:]
    fram3=data[2,:,:]
    fram2[:,0]=fram2[:,0]+75
    fram2[:,1]=fram2[:,1]+75
    fram3[:,0]=fram3[:,0]+150
    fram3[:,1]=fram3[:,1]+150

    for i in range(len(fram)):
        ax.plot([fram[i,0], fram2[i,0]], [fram[i,1], fram2[i,1]], "slateblue",alpha=0.1)
    for i in range(len(fram)):
        ax.plot([fram3[i,0], fram2[i,0]], [fram3[i,1], fram2[i,1]], "slateblue",alpha=0.1)    
    for i in range(len(edges[0])):
        ax.plot([fram[edges[0][i],0].item(), (fram[edges[1][i],0]).item()], [fram[edges[0][i],1].item(), (fram[edges[1][i],1]).item()], "slateblue",alpha=0.4)
    for i in range(len(edges[0])):
        ax.plot([fram2[edges[0][i],0].item(), (fram2[edges[1][i],0]).item()], [fram2[edges[0][i],1].item(), (fram2[edges[1][i],1]).item()], "slateblue",alpha=0.1)
    for i in range(len(edges[0])):
        ax.plot([fram3[edges[0][i],0].item(), (fram3[edges[1][i],0]).item()], [fram3[edges[0][i],1].item(), (fram3[edges[1][i],1]).item()], "slateblue",alpha=0.1)
                   
    ax.scatter(fram[:,0], fram[:,1],alpha=1, c="slateblue",s=10)

    ax.scatter(fram2[:,0], fram2[:,1],alpha=0.8, c="slateblue",s=10)
    ax.scatter(fram3[:,0], fram3[:,1],alpha=0.6, c="slateblue",s=10)
    fig.canvas.draw()
   
def visualize_cm(confusion_matrices,path_vis=None,time_steps=None):
    figures= [] # for storing the generated images
    for i,cm_k in enumerate(confusion_matrices):
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_k)
        disp.plot(ax=ax)
        ax.set_title("Epoche: {}".format(i))
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        figures.append(image)       
    imageio.mimsave(path_vis+'cm.gif', figures, duration=100)

def visualize_one_cm(cm,path_vis=None,time_steps=None,title="Confusion_matrix"):
    wandb.init()
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    ax.set_title(title)
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    image_wand = wandb.Image(image, caption=title)
    wandb.log({title: image_wand})     

def visualize_Adjacencey_matrix_from_edges(edges):
    edges_matrix=to_undirected(torch.tensor(edges),num_nodes=51)
    edges_matrix=to_dense_adj(torch.tensor(edges_matrix),max_num_nodes=51)
    edges_matrix=torch.squeeze(edges_matrix)
    plt.imshow(edges_matrix, cmap='Purples')
    plt.colorbar()
    plt.show()  
def visualize_adjacency_matrix(a_matrix):
    plt.imshow(a_matrix, cmap='gray')
    plt.colorbar()
    plt.show() 
def visualize_sample_withAdjacency(data,edges,vis_index=False):
        fig = plt.figure()
        ax = fig.add_subplot()
        fram=data
        print(fram.shape)
        ax.scatter(fram[:,0], fram[:,1],alpha=0.7, c="blue")
        if vis_index:
            for index in range(len(fram)):
                ax.annotate(index,(fram[index][0],fram[index][1])) 
        
        for i in range(51):
            for j in range(51):
                if edges[i][j]!=0:
                    alpha=edges[i][j]
                    ax.plot([fram[i][0],fram[j][0]],[fram[i][1],fram[j][1]],"blue",alpha=alpha)
def visualize_sample_withNode_wieghts(data,vis_index=False):
    
        #sns.color_palette("YlOrBr", as_cmap=True)
        sns.scatterplot(x="x", y="y", data=data,size='wieghts',hue='wieghts',palette="crest")
        #plt.title('Fig.No. 3: BUBBLE CHART')
       # plt.xlabel('Technical')
       # plt.ylabel('Job Proficiency') 
       # ax.scatter(fram[:,0], fram[:,1],alpha=0.7, c="blue",s=node_wieghts)
        
#%%
#path=parent_folder+"/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/08-23-16:47/"
name_file = 'minidata' #'minidata_mediapipe'
config_file=open(parent_folder+"config/"+name_file+".yml", 'r')
config = yaml.safe_load(config_file)
data_path=parent_folder+config['data_path']
labels_path=parent_folder+config['labels_path']
edges_path=parent_folder+config['edges_path']
idx_train= parent_folder+config['idx_train']
idx_test=parent_folder+config['idx_test']
edges=np.load(edges_path)
TS=config['TS']
num_nodes= config['n_joints']
#transform=RandomApply([RandomChoice([Rotate(),FlipV(),Compose([FlipV(),Rotate()])])],p=0.01)
#transform=RandomApply([Rotate()],p=0)
#transform=RandomApply([FlipV(),Rotate()],p=0.5)
transform=None
#%%
train_dataset=DataLoader(data_path,
                        labels_path,
                        edges_path,
                        idx_path=idx_train,
                        num_features=6,
                        num_nodes=num_nodes,
                        reshape_data=False,
                        expand_dim=False,
                        normalize_labels=False,
                        transform=transform,
                        maxMinNormalization=True)
#%%
#data=np.zeros((20,6,137,51,1))
data=np.zeros((20,77,num_nodes,6))
labels=[]
for i,sample in enumerate(train_dataset):
    data[i]=sample[0]
    labels.append(sample[1])
print(data.shape,len(labels))
#%%
print(data[0,0,0,:])
#%%
np.save("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/minidata/data.npy",data)
#%%
data=np.load("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/mediapipe/dataset_mediapipe.npy")
print(data.shape)
#%%
reshaped_tensor = np.transpose(data, (0, 2, 3, 1,4))  # Transpose dimensions from 8700,137,469,4) to 8600, 469, 4, 137
features= np.reshape(reshaped_tensor, (20, 137,51,6,1)).squeeze(axis=-1) 
print(features.shape)

#%%
edges=np.load("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/NoNose_edges.npy")

#%%
features=data
visualize_landmarks(features[:,:,:,:2],labels,edges,time_steps=10,vis_index=True,vis_edges=False)
#%%
features=data
features=np.concatenate((features[:,:,:11,:],features[:,:,19:,:]),axis=2)

visualize_sample(features[0],labels[0],edges,time_steps=1,vis_index=True,vis_edges=True)
#%%
#Half face
#idx=[5:14] [16:19] [25:31], [34:41], [45:50]
half_nodes= np.concatenate((features[0][:,5:14,:],features[0][:,16:19,:],features[0][:,25:31,:],features[0][:,34:41,:],features[0][:,45:50,:]),axis=1)

#%%
print(half_nodes.shape)
#%%
#get halfnodes edges
half_edges=get_edges(half_nodes[0][:,:2],"/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/edges_halfnodes.npy")
#%%

visualize_sample(half_nodes,labels[0],half_edges,time_steps=1,vis_index=True,vis_edges=True)

#%%
master_nods_edges=[[12 for x in range(0,51) ],[y for y in range(0,51)]]
#%%
np.array(master_nods_edges)
merged=np.hstack((edges,np.array(master_nods_edges)))
visualize_Adjacencey_matrix_from_edges(merged)
#%%
print(min(edges[0]))
#%%

#%%


#%%
np.save("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/edges_masternode.npy",merged)
#%%
visualize_temporal_edges(data[0])
#%%
adaptive=np.load("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/process_data/Adaptive_matrix.npy")

# %%
np.sum(adaptive[0][0])
# %%
adaptive[0].shape
# %%
plt.imshow((adaptive[0]+1)/2, cmap='gray')
plt.colorbar()
plt.show() 

# %%
plt.imshow(adaptive[1], cmap='gray')
plt.colorbar()
plt.show() 
# %%
normalizied =(adaptive[0]+1)/2
normalizied[0]
# %%
#obtain_normalized matrix with edges have wieghts over 70,
for i in range(51):
    for j in range(51):
        if normalizied[i][j]<0.7:
            normalizied[i][j]=0
        

# %%
plt.imshow(normalizied, cmap='gray')
plt.colorbar()
plt.show() 
# %%
visualize_sample_withAdjacency(data[0][0],normalizied,vis_index=True)
# %%
attention=np.load("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/process_data/Attention_nodes_l2.npy")

# %%
w=np.squeeze(attention[0])*10
frame=data[0][0]
x=frame[:,0]
y=frame[:,1]
x.shape
# %%
df=pd.DataFrame({"x":x,"y":y,"wieghts":w})
df.head()

#%%
visualize_sample_withNode_wieghts(df)
# %%
# %%

import seaborn as sns

# Your confusion matrix
cm = np.array([[323 ,  69],
 [130 ,233]])

# Convert to percentages
cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100

# Plotting the confusion matrix with percentages
plt.figure(figsize=(10, 7))
sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Percentage (%)'})

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Percentages (Binary)')
plt.show()


# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
# Base path where your confusion matrices are stored
base_path = "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/PartA/loso_ME87_new/"

# List to hold the confusion matrices
confusion_matrices = []

# Regular expression to extract confusion matrix from the log file
start_cm_pattern = re.compile(r"cm:\[\[(.*)")
# List to hold confusion matrices
confusion_matrices = []

# Iterate through the folders (assuming folder names are '1', '2', ..., '87')
for subject_id in range(0, 87):
    log_file_path = os.path.join(base_path, f"{subject_id}/1s+15k+multi+loso_ME87_test{subject_id}/log.txt")
    num_rows=1
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as file:
            lines = file.readlines()
            matrix_found = False
            cm = []
# Process the file line by line
            for line in lines:
                # Search for the start of the confusion matrix
                if not matrix_found:
                    match = start_cm_pattern.search(line)
                    if match:
                        row_str = match.group(1).strip()  # The first row
                        row_str = row_str.replace(']', '').replace('[', '')  # Remove remaining brackets
                        row = list(map(int, row_str.split()))
                        cm.append(row)
                        matrix_found = True
                else:
                    # Subsequent rows of the confusion matrix
                    num_rows=num_rows+1
                    if num_rows<=5:
                        row_str = line.strip().replace('[', '').replace(']', '')
                        if row_str:
                            row = list(map(int, row_str.split()))
                            cm.append(row)

            if cm:
                confusion_matrices.append(np.array(cm))
            else:
                print(f"No confusion matrix found in {log_file_path}")
    else:
        print(f"Log file not found: {log_file_path}")

# Stack all confusion matrices and calculate the mean
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

# Convert to percentage
mean_confusion_matrix_percent = mean_confusion_matrix / np.sum(mean_confusion_matrix, axis=1, keepdims=True) * 100

# Visualize the mean confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(mean_confusion_matrix_percent, annot=True, fmt=".2f", cmap="Blues", cbar=True)
plt.title("Mean Confusion Matrix (Percentage)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# %%

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the base path for the folders
base_path = "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/PartA/loso_ME87_new/"

# A pattern to detect the confusion matrix in the logs
cm_pattern = re.compile(r"cm:\[\[([0-9\s]+)\]\]")

# List to hold confusion matrices
confusion_matrices = []

# Iterate through the folders (assuming folder names are '1', '2', ..., '87')
for subject_id in range(0, 67):
    log_file_path = os.path.join(base_path, f"{subject_id}/1s+15k+multi+loso_LE67_test{subject_id}/log.txt")
    
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as file:
            log_content = file.read()
            matches = cm_pattern.findall(log_content)

            # If matches are found, we take the last one
            if matches:
                last_cm_str = matches[-1].strip()  # Get the last confusion matrix
                # Split rows by handling spaces and remove unwanted brackets
                cm = np.array([list(map(int, row.split())) for row in last_cm_str.split('] [')])
                confusion_matrices.append(cm)
            else:
                print(f"No confusion matrix found in {log_file_path}")
    else:
        print(f"Log file not found: {log_file_path}")

# Check if confusion matrices were found
if len(confusion_matrices) > 0:
    # Calculate the mean confusion matrix
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

    # Convert the mean confusion matrix to percentage
    mean_confusion_matrix_percent = mean_confusion_matrix / np.sum(mean_confusion_matrix, axis=1, keepdims=True) * 100

    # Visualize the mean confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_confusion_matrix_percent, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title("Mean Confusion Matrix (Percentage)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
else:
    print("No confusion matrices were found.")

# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample confusion matrices (binary and multi-class)
binary_cm = np.array([[82.4, 17.6], [35.81, 64.19]])
multi_cm = np.array([[10.20, 35.97, 39.80, 11.99, 2.04],
                     [6.98, 42.12, 34.88, 13.95, 2.07],
                     [5.08, 31.73, 37.82, 20.81, 4.57],
                     [3.70, 25.13, 33.07, 26.72, 11.38],
                     [3.58, 14.33, 29.48, 27.55, 25.07]])

# Create subplots for side-by-side visualizations
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Flip the binary confusion matrix
sns.heatmap(binary_cm, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Percentage (%)'}, ax=axes[0])
axes[0].set_title('Confusion Matrix with Percentages (Binary)')
axes[0].invert_yaxis()  # Flip the y-axis so label 0 is at the bottom

# Flip the multi-class confusion matrix
sns.heatmap(multi_cm, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Percentage (%)'}, ax=axes[1])
axes[1].set_title('Confusion Matrix with Percentages (Multi)')
axes[1].invert_yaxis()  # Flip the y-axis so label 0 is at the bottom

# Set common labels for both heatmaps
for ax in axes:
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()
# %%
