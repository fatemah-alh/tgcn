
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
from torch_geometric.utils import add_self_loops,to_undirected,to_dense_adj

parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)

from dataloader import DataLoader,Rotate,FlipV
from torchvision.transforms import RandomApply,RandomChoice,Compose
import wandb
from utiles import rotation_matrix_2d

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
        if vis_index:
            for index in range(len(fram)):
                ax.annotate(index,(fram[index][0],fram[index][1]))
        if vis=="2d":
            ax.scatter(fram[:,0], fram[:,1],alpha=0.7, c="blue")
            
        else:
            ax.scatter(fram[:,0], fram[:,1],fram[:,2], c="blue")
            ax.view_init()
        
        if vis_edges:
            for i in range(len(edges[0])):
                ax.plot([fram[edges[0][i],0].item(), (fram[edges[1][i],0]).item()], [fram[edges[0][i],1].item(), (fram[edges[1][i],1]).item()], "blue",alpha=0.3)
            
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        figures.append(image)
    if save:
        imageio.mimsave(path_vis+'vis_landmarks_3D_openface_absolut_eigenvectors_without_contur.gif', figures, duration=50)


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

def visualize_Adjacencey_matrix(edges):
    edges_matrix=to_undirected(torch.tensor(edges),num_nodes=51)
    edges_matrix=to_dense_adj(torch.tensor(edges_matrix),max_num_nodes=51)
    edges_matrix=torch.squeeze(edges_matrix)
    plt.imshow(edges_matrix, cmap='gray')
    plt.colorbar()
    plt.show()  
#%%
path=parent_folder+"/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/08-23-16:47/"
name_file = 'minidata'
config_file=open(parent_folder+"config/"+name_file+".yml", 'r')
config = yaml.safe_load(config_file)
data_path=parent_folder+config['data_path']
labels_path=parent_folder+config['labels_path']
edges_path=parent_folder+config['edges_path']
idx_train= parent_folder+config['idx_train']
idx_test=parent_folder+config['idx_test']
edges=np.load(edges_path)
TS=config['TS']
#transform=RandomApply([RandomChoice([Rotate(),FlipV(),Compose([FlipV(),Rotate()])])],p=0.01)
#transform=RandomApply([Rotate()],p=0)
#transform=RandomApply([FlipV(),Rotate()],p=0.5)
transform=None
train_dataset=DataLoader(data_path,
                        labels_path,
                        edges_path,
                        idx_path=idx_train,
                        num_features=6,
                        num_nodes=51,
                        reshape_data=False,
                        expand_dim=False,
                        normalize_labels=False,
                        transform=transform)

#data=np.zeros((20,6,137,51,1))
data=np.zeros((20,137,51,6))
labels=[]
for i,sample in enumerate(train_dataset):
    data[i]=sample[0]
    labels.append(sample[1])
print(data.shape,len(labels))
#%%
reshaped_tensor = np.transpose(data, (0, 2, 3, 1,4))  # Transpose dimensions from 8700,137,469,4) to 8600, 469, 4, 137
features= np.reshape(reshaped_tensor, (20, 137,51,6,1)).squeeze(axis=-1) 
print(features.shape)


visualize_landmarks(features[:,:,:,:2],labels,edges,time_steps=1,vis_index=True,vis_edges=True)
#visualize_sample(features[0],labels[0],edges,time_steps=1,vis_index=True,vis_edges=True)
