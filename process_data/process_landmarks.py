#%%
import numpy as np
import pandas as pd
import yaml
import sys
parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)
from utiles import get_file_names
from tqdm import tqdm 
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import torch
from torch_geometric.utils import add_self_loops,is_undirected,to_undirected,contains_self_loops
from sklearn.preprocessing import MinMaxScaler

"""
This file should be run after the run of the file "extract_3DLandmarks_openFace.py"
it will process the data in landmarks folder and create a uninqe dataset file in numpy format
the data set will have the seguent dim:
(num_samples, num_frames,num_landmarks,num_coordinates)

This will perform :
1- aligemnet of landmarks to have frontal position.
2- standarize data to have zero means and unitary variance
3- normalize dividing by frobinus norm.
4- calculate velocity on x,y
5- Insert centroid of landmarls before alignement, to account of head movement
"""
def visualize_landmarks(data,label_data,edges=[],time_steps=137,vis_edges=False,vis="2d",vis_index=False,save=False):
    """
    this function will visualize landmarks of data tensor, just the f
    """
    
    figures=[]
    origin=[0,0,0]
    for time_step in tqdm(range(time_steps)):
        # Create a figure and axes for subplots
        if vis=="2d":
            fig, axs = plt.subplots(5, 4, figsize=(30, 30))
        else:
            fig, axs = plt.subplots(5, 4, figsize=(30, 30),subplot_kw=dict(projection='3d'))
        
        # Flatten the axes array
        axs = axs.flatten()
        for d in range(0,20):
            fram=data[d,time_step,:,:]
            axs[d].set_title(f"VAS level: {label_data[d]}")
            if vis_index:
                for index in range(len(fram)):
                    axs[d].annotate(index,(fram[index][0],fram[index][1]))
            if vis=="2d":
                axs[d].scatter(fram[:,0], fram[:,1], alpha=0.8)
            else:
                axs[d].scatter(fram[:,0], fram[:,1],fram[:,2], alpha=0.8)
                #axs[d].quiver(origin[0],origin[1],origin[2], autovettori[0,0],autovettori[1,0],autovettori[2,0],color="r")
                #axs[d].quiver(origin[0],origin[1],origin[2], autovettori[0,1],autovettori[1,1],autovettori[2,1],color="g")
                #axs[d].quiver(origin[0],origin[1],origin[2], autovettori[0,2],autovettori[1,2],autovettori[2,2],color="b")
                #axs[d].quiver(origin[0],origin[1],origin[2], n[0],n[1],n[2],color="fuchsia")
                axs[d].quiver(origin[0],origin[1],origin[2],0,0,1,color="y")
                axs[d].text(0,0,1,"z")
                axs[d].quiver(origin[0],origin[1],origin[2], 0,1,0,color="y")
                axs[d].text(0,1,0,"y")
                axs[d].quiver(origin[0],origin[1],origin[2], 1,0,0,color="y")
                axs[d].text(1,0,0,"x")
                #axs[d].view_init(-30,60)
            
            if vis_edges:
                axs[d].plot([fram[edges[0],0],fram[edges[1],0]],[fram[edges[0],1],fram[edges[1],1]],"blue")
            
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        figures.append(image)
    if save:
        imageio.mimsave(path_vis+'vis_landmarks_3D_openface_absolut_eigenvectors_without_contur.gif', figures, duration=50)
def save_labels(csv_file,label_data):
    df = pd.read_csv(csv_file,sep='\t')
    labels=df['class_id'].values
    np.save(label_data,labels)
    return labels

def plot_histogram(values):
    q25, q75 = np.percentile(values, [25, 75])
    bin_width = 2 * (q75 - q25) * len(values) ** (-1/3)
    print(np.max(values),np.min(values),bin_width)
    bins = round((values.max() - values.min()) / bin_width)
    plt.hist(values,range=(np.min(values),np.max(values)),bins=bins)

def standarization_train(data_train,num_features,calc_std=True):
    means=[]
    stds=[]
    for i in range(0,num_features):
        mean=np.mean(data_train[:,:,:,i])
        std=np.std(data_train[:,:,:,i])
        
        data_train[:,:,:,i]=(data_train[:,:,:,i]-mean)
        if calc_std:
           data_train[:,:,:,i]= data_train[:,:,:,i]/std

        means.append(mean)
        stds.append(std)
    return data_train,means,stds
def standarization_test(data_test,num_features,means,stds):
    for i in range(0,num_features):
        data_test[:,:,:,i]=(data_test[:,:,:,i]-means[i])/stds[i]
    return data_test

def split_idx_train_test(idx_train_path,idx_test_path,csv_file,filter_90=None):
    low_expressiv_ids=["082315_w_60", "082414_m_64", "082909_m_47","083009_w_42", "083013_w_47", 
                        "083109_m_60", "083114_w_55", "091914_m_46", "092009_m_54","092014_m_56", 
                        "092509_w_51", "092714_m_64", "100514_w_51", "100914_m_39", "101114_w_37", 
                        "101209_w_61", "101809_m_59", "101916_m_40", "111313_m_64", "120614_w_61"]
    validation_subjects_id=["100914_m_39", "101114_w_37", "082315_w_60", "083114_w_55", 
                            "083109_m_60", "072514_m_27", "080309_m_29", "112016_m_25", 
                            "112310_m_20", "092813_w_24", "112809_w_23", "112909_w_20", 
                            "071313_m_41", "101309_m_48", "101609_m_36", "091809_w_43", 
                            "102214_w_36", "102316_w_50", "112009_w_43", "101814_m_58", 
                            "101908_m_61", "102309_m_61", "112209_m_51", "112610_w_60", 
                            "112914_w_51", "120514_w_56"]
    df = pd.read_csv(csv_file,sep='\t')
    
    mask = df['subject_name'].isin(low_expressiv_ids)
    idx_low= df.loc[mask].index.tolist()
    print(len(idx_low))
    if(filter_90!=None):
        idx_filter_90=np.load(filter_90)
    else:
        idx_filter_90=[]
    subject_name=df['subject_name'].to_numpy()

    idx_test=[]
    idx_train=[]
    for i in range(len(subject_name)):
        test=False
        if i not in idx_filter_90 and i not in idx_low:
            for j in validation_subjects_id:
                if subject_name[i].strip()==j:
                    test=True
            if test:
                idx_test.append(i)        
            else:
                idx_train.append(i)

    print("Percent test",len(idx_test)/(len(idx_test)+len(idx_train)))
    print("len test: ", len(idx_test),"len train:",len(idx_train),"total:",len(idx_test)+len(idx_train))
    #assert len(idx_test)+len(idx_train)==len(subject_name)-len(idx_filter_90)-len(idx_low)

    np.save(idx_train_path,idx_train)
    np.save(idx_test_path,idx_test)
    return idx_train,idx_test

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
    
def delete_contour(sample):
    sample=sample[17:,:]
    return sample
def frobenius_norm(arr):
    """
    frame: array
    """
    norm=np.linalg.norm(arr,ord= 'fro')
    if norm!=0:
        arr/norm
    return arr
def standardize(frame):
    return (frame - np.mean(frame ,axis=0)) /np.std(frame,axis=0)
def center_coordinate(frame):
    x_center=np.mean(frame[:,0])
    y_center=np.mean(frame[:,1])
    z_center=np.mean(frame[:,2])
   # x_std=np.std(frame[:,0])
   # y_std=np.std(frame[:,0])
   # z_std=np.std(frame[:,0])
    frame[:,0]=frame[:,0]-x_center
    frame[:,1]=frame[:,1]-y_center
    frame[:,2]=frame[:,2]-z_center
    return frame,np.array([[x_center,y_center,z_center]])
def flip_y_coordiante(frame):
    frame[:,1]=-frame[:,1]
    return frame
def get_rotation_matrix(lndk_centered):
  e_vals_g, e_vect_g = np.linalg.eig(np.matmul(lndk_centered.T, lndk_centered))
  e_vals_g = np.real(e_vals_g)
  e_vect_g = np.real(e_vect_g)
  # The eigenvectors are the columns of e_vect_g
  #print(e_vals_g)
  #print(e_vect_g)
  #print('...done!')
  # identify the eigenvector that is most closely alligned to the x axis
  idx = np.abs(e_vect_g[0,:]).argmax()
  # normalize it and use it to define the first row of the rotation matrix
  # the sign function is used to set the same orientation as the x axis
  v = e_vect_g[:,idx]*np.sign(e_vect_g[0,idx])
  v = v / np.linalg.norm(v)
  base_m = v.reshape(1,-1)

  # now identify the eigenvector that is most closely alligned to the y axis
  idx = np.abs(e_vect_g[1,:]).argmax()
  # normalize it and use it to define the second row of the rotation matrix
  # the sign function is used to set the same orientation as the y axis
  v = e_vect_g[:,idx]*np.sign(e_vect_g[1,idx])
  v = v / np.linalg.norm(v)
  base_m = np.vstack( (base_m, v.reshape(1,-1)) )

  # now identify the eigenvector that is most closely alligned to the z axis
  idx = np.abs(e_vect_g[2,:]).argmax()
  # normalize it and use it to define the second row of the rotation matrix
  # the sign function is used to set the same orientation as the z axis
  v = e_vect_g[:,idx]*np.sign(e_vect_g[2,idx])
  v = v / np.linalg.norm(v)
  base_m = np.vstack( (base_m, v.reshape(1,-1)) )
  return base_m
def calc_velocity(sample):
    velcetiy=[]
    for i in range(1,len(sample)):
        velcetiy.append(sample[i]-sample[i-1])
    return velcetiy  
def add_centroid(frame,centroid):
   
    frame=np.append(frame,centroid,axis=0)
    return frame
def preprocess_frame(frame):
    frame=delete_contour(frame)
    frame=flip_y_coordiante(frame)
    frame,centroid=center_coordinate(frame)
    R_matrix=get_rotation_matrix(frame)
    frame=np.matmul( R_matrix, frame.T ).T
    frame=add_centroid(frame,centroid)
    print(frame.shape)
    #frame=standardize(frame)
    #frame=frobenius_norm(frame)
    return frame
def process_all_data(landmarks_folder:str,filesnames:list,normalized_data:np.array,path_save:str):
    """
    Iterate over all landmarks.npy for each sample, apply
    1- make coordiates have zero mean 
    2- Divise each frame by frobenius norm
    3-calculate centroid???
    4 remove rotaione of the face, align points to be in frontal face position. 
    5- calculate velocitie
    create the data matrix that containes all samples [num_samples,num_frame,num_landmarks,num_featuers]
    """
    
    for i in tqdm(range (0,len(filesnames))):
        path=landmarks_folder+filesnames[i]+"/"+filesnames[i].split("/")[1]+".npy"
        sample=np.load(path) #[138,68,3] 
        processed_sample=np.zeros((138,52,3))
        for j in range(len(sample)):
            processed_sample[j]=preprocess_frame(sample[j])
           
        velocity=calc_velocity(processed_sample)
        data=np.concatenate((processed_sample[:-1,:,:], velocity), axis=2) 
        normalized_data[i][:data.shape[0]]= data
    print(normalized_data[:,:,:,0].shape,np.max(normalized_data[:,:,:,0]),np.min(normalized_data[:,:,:,0]),np.max(normalized_data[:,:,:,1]),np.min(normalized_data[:,:,:,1]),np.max(normalized_data[:,:,:,2]),np.min(normalized_data[:,:,:,2]))
    print("Contains Nan values",np.isnan(normalized_data).any())
    #normalized_data=np.nan_to_num(normalized_data)        
    np.save(path_save,normalized_data)
    return normalized_data


name_exp = 'open_face'
    
config_file=open(parent_folder+"config/"+name_exp+".yml", 'r')
path_vis=parent_folder+"/data/PartA/vis/" # path to save gif of visualizzation
name_file = 'open_face' # name of conficuration file.

config = yaml.safe_load(config_file)
labels_path=parent_folder+config['labels_path']
idx_train= parent_folder+config['idx_train']
video_folder_path=parent_folder+config["video_path"]
landmarks_path=parent_folder+config["landmarks_path"]
csv_file=parent_folder+config["csv_file"]
data_path=parent_folder+config["data_path"]
edges_path=parent_folder+config["edges_path"]
idx_train=parent_folder+config["idx_train"]
idx_test=parent_folder+config["idx_test"]

filter_idx_90=parent_folder+config["filter_idx_90"]
#print frequencies of labels

filesnames=get_file_names(csv_file)

if name_file=="mediapipe":
    normalized_data = np.zeros((8700, 137, 469, 4), dtype=np.float32)
    

if name_file=="open_face":
    normalized_data = np.zeros((8700, 137, 52, 6), dtype=np.float32)

#%%

#%%   #Create the the dataset file with process
#normalized_data=process_all_data(landmarks_path,filesnames,normalized_data,data_path)
#print(normalized_data[:,:,:,0].shape,np.max(normalized_data[:,:,:,0]),np.min(normalized_data[:,:,:,0]),np.max(normalized_data[:,:,:,1]),np.min(normalized_data[:,:,:,1]),np.max(normalized_data[:,:,:,2]),np.min(normalized_data[:,:,:,2]))
    
#%%   # create label file
#labels=save_labels(csv_file,labels_path)
labels=np.load(labels_path)
print(labels)


#%%   #Split to train set and test 
#split_idx_train_test(idx_train,idx_test,csv_file,filter_idx_90)

#%%   #Create the edges array 
data=np.load(data_path)
idx_train_=np.load(idx_train)
idx_test_=np.load(idx_test)
#%%

#%%
#data=np.nan_to_num(data)
#%%
#print(data[idx_test_,:,:,0].shape,np.max(data[idx_test_,:,:,0]),np.min(data[idx_test_,:,:,0]),np.max(data[idx_test_,:,:,1]),np.min(data[idx_test_,:,:,1]),np.max(data[idx_test_,:,:,2]),np.min(data[idx_test_,:,:,2]),np.max(data[idx_test_,:,:,3]),np.min(data[idx_test_,:,:,3]),np.max(data[idx_test_,:,:,4]),np.min(data[idx_test_,:,:,4]),np.max(data[idx_test_,:,:,5]),np.min(data[idx_test_,:,:,5]))
#print(data[idx_train_,:,:,0].shape,np.max(data[idx_train_,:,:,0]),np.min(data[idx_train_,:,:,0]),np.max(data[idx_train_,:,:,1]),np.min(data[idx_train_,:,:,1]),np.max(data[idx_train_,:,:,2]),np.min(data[idx_train_,:,:,2]),np.max(data[idx_train_,:,:,3]),np.min(data[idx_train_,:,:,3]),np.max(data[idx_train_,:,:,4]),np.min(data[idx_train_,:,:,4]),np.max(data[idx_train_,:,:,5]),np.min(data[idx_train_,:,:,5]))
#print(np.mean(data[idx_test_,:,:,0]),np.mean(data[idx_test_,:,:,1]),np.mean(data[idx_test_,:,:,2]),np.mean(data[idx_test_,:,:,3]),np.mean(data[idx_test_,:,:,4]),np.mean(data[idx_test_,:,:,5]))
#print(np.mean(data[idx_train_,:,:,0]),np.mean(data[idx_train_,:,:,1]),np.mean(data[idx_train_,:,:,2]),np.mean(data[idx_train_,:,:,3]),np.mean(data[idx_train_,:,:,4]),np.mean(data[idx_train_,:,:,5]))
#%%

#%%
data_train=data[idx_train_,:,:,:]
data_test=data[idx_test_,:,:,:]
#%%
"""
standard_data,means,stds=standarization_train(data_train,6)
print(means,stds)
standard_data_test=standarization_test(data_test,6,means,stds)
data[idx_train_,:,:,:]=standard_data
data[idx_test_,:,:,:]=standard_data_test
#np.save("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/openFace/dataset_openFace_standarized.npy",data)
"""
#%%
#%%
x_values=data[idx_train_,:,:,1].flatten()

#plot_histogram(x_values)


#%%
#edges_index=get_edges(data[0,0,:51,:2],edges_path)


#%%
edges_index=np.load(edges_path)
#%% #visualize
#visualize_landmarks(data[100:400],labels,edges_index,vis_edges=True,time_steps=10)

#%%

def split_all_partecipant(idx_train_path,idx_test_path,csv_file):
    low_expressiv_ids=["082315_w_60", "082414_m_64", "082909_m_47","083009_w_42", "083013_w_47", 
                        "083109_m_60", "083114_w_55", "091914_m_46", "092009_m_54","092014_m_56", 
                        "092509_w_51", "092714_m_64", "100514_w_51", "100914_m_39", "101114_w_37", 
                        "101209_w_61", "101809_m_59", "101916_m_40", "111313_m_64", "120614_w_61"]
    df = pd.read_csv(csv_file,sep='\t')
    
    mask = df['subject_name'].isin(low_expressiv_ids)
    idx_low= df.loc[mask].index.tolist()
    print(len(idx_low))
    subject_name=df['subject_name'].to_numpy()
    print(subject_name.shape)
    idx_test=[]
    idx_train=[]
    pass
#split_all_partecipant(idx_train,idx_test,csv_file)

def min_max_normalize(frame):

    pass