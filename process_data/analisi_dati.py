#%%
import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys
from sklearn.manifold import TSNE

parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)
from main import Trainer
path_save=parent_folder+"data/PartA/vis/"
name_file = 'open_face'
config_file=open(parent_folder+"config/"+name_file+".yml", 'r')
config = yaml.safe_load(config_file)
path_pretrained_model="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/08-16-15:53/best_model.pkl"

def plot_t_sne(config,path_pretrained_model,vis="2d",path_save=path_save,classes_option="all"):
    cdict = {0:'green', 1: 'blue', 2: 'gold', 3:'orange',4:'red'}
    trainer=Trainer(config=config)
    embeddings,classes,predicted=trainer.get_embedding(path_pretrained_model)
    if classes_option=="all":
        filterd_classes=classes
        filtred_embedings=embeddings
    else:
        filtred_idx=np.where(classes=classes_option)
        filterd_classes=classes[filtred_idx]
        filtred_embedings=embeddings[filtred_idx]
    tsn_embedded = TSNE(n_components=3, 
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
            ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1], c = cdict[g], label = g)
        else:
            ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1],tsn_embedded[ix,2], c = cdict[g], label = g)
        # ax.view_init(-30,60)

    ax.legend()
    ax.set_title(f"Embeddings visualising {classes_option} ")
    plt.show()
    plt.savefig(path_save+"tsn.png")



#%%
"""
g=2
fig, ax = plt.subplots()
ix = np.where(classes == g)
ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1], c = cdict[g], label = g)
ax.legend()
ax.set_title(f"Embeddings visualising ")
plt.show()

"""