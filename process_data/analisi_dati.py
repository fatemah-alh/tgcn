#%%
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.manifold import TSNE
parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)
from main import Trainer
from helper.Config import Config
from helper.DataHandler import DataHandler
from helper.Logger import Logger
from helper. Evaluation import Evaluation

config_file="open_face"
parent_folder="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
config =Config.load_from_file(parent_folder+"/config/"+config_file+".yml")

def get_all_outputs(config,path_pretrained_model):
    trainer=Trainer(config=config)
   # trainer.init_trainer()
    return trainer.get_all_outputs(path=path_pretrained_model)
def get_true_predicted(targets,predicted,outputs):
    print(outputs[0].shape)
    #outputs=outputs.view(1914,-1)
    idx_0_0=np.where((targets==0) & (predicted==0))[0]
    idx_4_4=np.where((targets==4) & (predicted==4))[0] 
    print(idx_0_0)
    outputs_0=outputs[idx_0_0]
    outputs_4=outputs[idx_4_4]
    print(outputs_4[0][:-1])
    plt.plot(outputs_4[0][:-1],label= f"Label: 4")
    plt.plot(outputs_0[0][:-1],label= f"Label: 0")
    plt.set_title("Outputs of the network at each block")
    plt.xlabel("Time Step")
    plt.ylabel("VAS")
    plt.legend()
    plt.show()
    pass

def get_embeddings(config,path_pretrained_model):
    trainer=Trainer(config=config)
    return trainer.get_embedding(path_pretrained_model)

def get_TSNE(embeddings,
             classes,
             predicted,
             predicted_label=4,
             vis="2d",
             title="Embeddings visualising predicted 4",
             path_save=parent_folder+"data/PartA/vis/"):
    
    labels=np.stack((classes,predicted),axis=1)
    pain_class=1 #4
    
    if predicted_label==4:
        idx_0_4=np.where((labels[:,0]==0) & (labels[:,1]==pain_class))[0] #Red
        idx_4_4=np.where((labels[:,0]==pain_class) & (labels[:,1]==pain_class))[0] #Green
        labels_4=np.concatenate((idx_0_4,idx_4_4))
        filtred_idx=labels_4
        filterd_classes=classes[filtred_idx]
        filtred_embedings=embeddings[filtred_idx]
        cdict = {0:'red',pain_class:'green'}
    elif predicted_label==0:
        idx_4_0=np.where((labels[:,0]==pain_class) & (labels[:,1]==0))[0] #Red
        idx_0_0=np.where((labels[:,0]==0) & (labels[:,1]==0))[0] #Green
        labels_0=np.concatenate((idx_4_0,idx_0_0))
        filtred_idx=labels_0
        filterd_classes=classes[filtred_idx]
        filtred_embedings=embeddings[filtred_idx]
        cdict = {0:'green',pain_class:'red'}

    elif predicted_label=="all":
        filterd_classes=classes
        filtred_embedings=embeddings
        cdict = {0:'green', 1: 'blue', 2: 'gold', 3:'orange',4:'red'}
    elif predicted_label=="true":
        idx_0_0=np.where((labels[:,0]==0) & (labels[:,1]==0))[0] #Green
        idx_4_4=np.where((labels[:,0]==pain_class) & (labels[:,1]==pain_class))[0] #Green
        labels_true=np.concatenate((idx_0_0,idx_4_4))
        filtred_idx=labels_true
        filterd_classes=classes[filtred_idx]
        filtred_embedings=embeddings[filtred_idx]
        cdict = {0:'blue',pain_class:'orange'}
    print(classes.shape,predicted.shape,labels.shape,embeddings.shape)
    #print(idx_0_4.shape,idx_4_0.shape,idx_0_0.shape,idx_4_4.shape,labels_0.shape,labels_4.shape)
    print(filterd_classes.shape,filtred_embedings.shape)

    tsn_embedded = TSNE(n_components=2, 
                        learning_rate='auto',
                        init='random',
                        perplexity=30).fit_transform(filtred_embedings)
    

    
    if vis=="2d":
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    for g in np.unique(filterd_classes):
        ix = np.where(filterd_classes == g)
        if vis=="2d":
            ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1], c = cdict[g], label = g,s=1)
        else:
            ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1],tsn_embedded[ix,2], c = cdict[g], label = g,s=1)
        # ax.view_init(-30,60)

    ax.legend()
    ax.set_title(title)
    plt.show()
    plt.savefig(path_save+title+".png")


def get_ids(labels,predicted=4,pain_class=1):
    idx_0_4=np.where((labels[:,0]==0) & (labels[:,1]==pain_class))[0]
    idx_4_0=np.where((labels[:,0]==pain_class) & (labels[:,1]==0))[0] #Red


#%%
targets,predicted,outputs=get_all_outputs(config,"/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/1s+15k+multi+MinMaxNorm_xy+NoLabelNorm/best_model.pkl")
#%%

#%%
idx_0_0=np.where((targets==0) & (predicted==0))[0]
idx_4_4=np.where((targets==4) & (predicted==4))[0] 
idx_3_3=np.where((targets==3) & (predicted==3))[0]
print(len(idx_0_0),len(idx_4_4))
#%%
outputs=np.array(outputs)

#%%
outputs_0=outputs[idx_0_0]
#%%
outputs_4=outputs[idx_4_4]
#%%
outputs_0=[row for row in outputs_0]
outputs_4=[row for row in outputs_4]
#%%

#%%
outputs_3=outputs[idx_3_3]
outputs_3=[row for row in outputs_3]

#%%
idx_0_4=np.where((targets==0) & (predicted==4))[0]
outputs_0_4=outputs[idx_0_4]
outputs_0_4=[row for row in outputs_0_4]
#%%
len(outputs_0_4)
#%%
for i in outputs_0_4[3:6]:
    plt.plot(i, color="b")
plt.title("Outputs of the network at each block __video label :3")
plt.xlabel("Time Step")
plt.ylabel("VAS")
plt.show()


#%%
outputs_4=outputs[idx_4_4]
#%%
get_true_predicted(targets,predicted,outputs)
#%%
embeddings,classes,predicted,initial_label= get_embeddings(config,
#path_pretrained_model="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/1s+15k+filterlow_react+binary_class_regression/best_model.pkl"
path_pretrained_model="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/1s+15k+filterlow_react+binary_class_regression+adaptiv/best_model.pkl"
   )
#%%
adaptive=np.load("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/process_data/Adaptive_matrix.npy")

#%%
adaptive.shape
#%%
get_TSNE(embeddings,
         classes,
         predicted,
         predicted_label="true",
         vis="2d",
         title="Embeddings visualising predicted true - Daptive model",
         path_save=parent_folder+"./data/PartA/vis/")

# %%