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
from helper.Evaluation import Evaluation

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

             lr=10,
             perp=30,
             early_exaggeration=6,
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
                        learning_rate=lr,#'auto',#[10,1000] #34,50max(N / early_exaggeration / 4, 50) where N is the sample size
                        init='random',
                        early_exaggeration=early_exaggeration,#defualt=12
                        #max_iter=2000,
                        n_iter_without_progress=1000,
                        n_iter=50000,
                        perplexity=perp).fit_transform(filtred_embedings)
    

    if vis=="2d":
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    """
    for g in np.unique(filterd_classes):
        ix = np.where(filterd_classes == g)
        if vis=="2d":
            ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1], c = cdict[g], label = g,s=1)
        else:
            ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1],tsn_embedded[ix,2], c = cdict[g], label = g,s=1)
        # ax.view_init(-30,60)
    
    ax.legend()
    ax.set_title(title+"True")
    plt.show()
   
    plt.savefig(path_save+title+str(lr)+str(perp)+str(early_exaggeration)+"_predicted.png")


    if vis=="2d":
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    
    for g in np.unique(predicted):
        ix = np.where(predicted == g)
        if vis=="2d":
            ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1], c = cdict[g], label = g,s=1)
        else:
            ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1],tsn_embedded[ix,2], c = cdict[g], label = g,s=1)
        # ax.view_init(-30,60)
    

    """
    
    idx_4_0=np.where((labels[:,0]==1) & (labels[:,1]==0))[0] 
    idx_0_0=np.where((labels[:,0]==0) & (labels[:,1]==0))[0] 
    idx_0_4=np.where((labels[:,0]==0) & (labels[:,1]==1))[0]
    idx_4_4=np.where((labels[:,0]==1) & (labels[:,1]==1))[0] 
    print(len(idx_4_0),len(idx_0_0),len(idx_0_4),len(idx_4_4))
    ax.scatter(tsn_embedded[idx_0_0,0], tsn_embedded[idx_0_0,1], c = "green", label = "00",s=1)
    ax.scatter(tsn_embedded[idx_4_4,0], tsn_embedded[idx_4_4,1], c = "blue", label = "11",s=1)
    ax.scatter(tsn_embedded[idx_4_0,0], tsn_embedded[idx_4_0,1], c = "red", label = "10",s=1)
    ax.scatter(tsn_embedded[idx_0_4,0], tsn_embedded[idx_0_4,1], c = "orange", label = "01",s=1)
   
      
    ax.legend()
    ax.set_title(title)
    plt.show()
   
    plt.savefig(path_save+title+str(lr)+str(perp)+str(early_exaggeration)+"_predicted.png")


def get_ids(labels,pain_class=1):
    idx_0_4=np.where((labels[:,0]==0) & (labels[:,1]==pain_class))[0]
    idx_4_0=np.where((labels[:,0]==pain_class) & (labels[:,1]==0))[0] #Red


#%%
def TSNE_last_frame():
    pass
def TSNE_match_output_withembedings():
    pass

#%%
pretrained="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/PartA/binary_custom_round/best_model.pkl"

#%%
targets,predicted,outputs=get_all_outputs(config,pretrained)
#%%
outputs.shape
#%%
#%%
embed_agcn,embed_gru,classes,predicted= get_embeddings(config,
path_pretrained_model=pretrained)
#%%
print(embed_agcn.shape,embed_gru.shape,classes.shape,predicted.shape)

last_embed_agcn=embed_agcn[:,-1,:]
last_embed_agcn.shape
#%%
last_embed_gru=embed_gru[:,-1,:]
last_embed_gru.shape
#%%
classes.shape
np.save("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/embedings/train_emb_gru.npy",last_embed_gru)
np.save("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/embedings/train_labels.npy",classes)
#%%

test_data=np.load("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/embedings/test_emb_gru.npy")
test_labels=np.load("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/embedings/test_labels.npy")

#%%
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=50, random_state=0,n_estimators=200)
clf.fit(last_embed_gru, classes)
pred_test=clf.predict(test_data)
acc=np.mean(pred_test == test_labels)
#%%
acc
#%%
get_TSNE(last_embed_agcn,
         classes,
         predicted,
         predicted_label="all",
         lr=100,
         perp=50,
         early_exaggeration=12,
         vis="2d",
         title="Embeddings_binary_agcn",
         path_save=parent_folder+"./data/PartA/vis/")
    

get_TSNE(last_embed_gru,
         classes,
         predicted,
         predicted_label="all",
         lr=100,
         perp=50,
         early_exaggeration=12,
         vis="2d",
         title="Embeddings_binary_gru",
         path_save=parent_folder+"./data/PartA/vis/")



#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=128)
reduced_last_embed_gru = pca.fit_transform(last_embed_gru)  # X is your 2000x7000 dataset


get_TSNE(reduced_last_embed_gru,
         classes,
         predicted,
         predicted_label="all",
         lr=400,
         perp=100,
         early_exaggeration=12,
         vis="2d",
         title="Embeddings_binary_gru",
         path_save=parent_folder+"./data/PartA/vis/")

#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=128)
reduced_last_embed_agcn = pca.fit_transform(last_embed_agcn)  # X is your 2000x7000 dataset

get_TSNE(reduced_last_embed_agcn,
         classes,
         predicted,
         predicted_label="all",
         lr=400,
         perp=100,
         early_exaggeration=12,
         vis="2d",
         title="Embeddings_binary_agcn",
         path_save=parent_folder+"./data/PartA/vis/")
    
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
#%%
adaptive=np.load("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/process_data/Adaptive_matrix.npy")

#%%
adaptive.shape
#%%
# %%