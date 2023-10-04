#%%
import torch
import wandb
import numpy as np
from itertools import chain
from models.aagcn import aagcn_network
from models.a3tgcn import A3TGCN2_network
from dataloader import DataLoader,Rotate,FlipV
from tqdm import tqdm 
import torch.optim.lr_scheduler as lr_scheduler
import datetime
import os
import yaml
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from sklearn.metrics import  ConfusionMatrixDisplay
from torcheval.metrics.functional import multiclass_f1_score
from torchvision.transforms import RandomApply,RandomChoice,Compose
#wandb.login() #just  for first run


class Trainer():
    def __init__(self, config) -> None:
        name_exp="open_face"
   
        self.parent_folder="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
        self.cm=[]
        self.config=config
        self.lr=config['lr']
        self.LOG_DIR=self.parent_folder+config['LOG_DIR']
        print(self.LOG_DIR)
        self.num_epoch=config['num_epoch']
        self.gpu=config['gpu']
        self.weight_decay=config['weight_decay']
        self.step_decay=config['step_decay']
        self.name_exp=config['name_exp']
        #model parameters
        self.num_features=config['num_features']
        self.embed_dim=config['embed_dim']
        
        self.TS=config['TS']
        self.continue_training=config['continue_training']
        self.pretrain_model=self.LOG_DIR+config['pretrain_model']
        self.model_name=config['model_name']
        #data set parametrs 
        self.data_path=self.parent_folder+config['data_path']
        self.labels_path=self.parent_folder+config['labels_path']
        self.edges_path=self.parent_folder+config['edges_path']
        self.idx_train=self.parent_folder+config['idx_train']
        self.idx_test=self.parent_folder+config['idx_test']
        self.batch_size=config['batch_size']
        self.num_nodes=config['n_joints']
        self.optimizer_name=config['optimizer']
        self.num_subset=config['num_subset']
        self.num_features=config['num_features']
        self.adaptive=config['adaptive']
        self.attention=config['attention']
        self.kernel_size=config['t_kernel_size']
        self.hidden_size=config['hidden_size']
        self.bn=config['bn']
        self.gru_layer=config['gru']
        self.strid=config["strid"]
        self.num_classes=config["num_classes"]
        self.augmentaion=config["augmentaion"]
        self.aug_type=config["Aug_type"]
        self.prop=config["prop"]
        if self.num_classes==2:
            self.classes=[0,1]
        elif self.num_classes==3:
            self.classes=[0,1,2]
        else:
            self.classes=[0,1,2,3,4]
       
        self.set_device()
        self.load_edges()
        self.load_datasets()
        self.load_model()
        self.load_optimizer()
        self.load_loss()
        print("Adaptive:",self.adaptive)
   
    def set_log_dir(self,name=None):
        self.name=name
        if self.name==None:
            self.name=datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.log_dir=os.path.join(self.LOG_DIR,self.name)
        os.makedirs(self.log_dir,exist_ok=True)
    def set_device(self):
        if torch.cuda.is_available():
            print("set cuda device")
            self.device="cuda"
            torch.cuda.set_device(self.gpu)
        else:
            self.device="cpu"
            print('Warning: Using CPU')
    def load_edges(self):
        self.edge_index=torch.LongTensor(np.load(self.edges_path)).to(self.device)
    def load_datasets(self):
        if self.augmentaion:
            if self.aug_type=="r":
                self.transform=RandomApply([Rotate()],p=self.prop)
                print("augmentaion rotation..")
            if self.aug_type=="f":
                self.transform=RandomApply([FlipV()],p=self.prop)
                print("augmentaion flip..")
            if self.aug_type=="r+f":
                self.transform=RandomApply([RandomChoice([Rotate(),FlipV(),Compose([FlipV(),Rotate()])])],p=self.prop)
            
                print("augmentaion rotaion + flip..")
        else:
            self.transform=None
            
        self.train_dataset=DataLoader(self.data_path,
                                      self.labels_path,
                                      self.edges_path,
                                      idx_path=self.idx_train,
                                      model_name=self.model_name,
                                      num_features= self.num_features,
                                      num_nodes=self.num_nodes,
                                      num_classes=self.num_classes,
                                      transform=self.transform)
        self.test_dataset=DataLoader(self.data_path,
                                     self.labels_path,
                                     self.edges_path,
                                     idx_path=self.idx_test,
                                     model_name=self.model_name,
                                     num_features= self.num_features,
                                     num_nodes=self.num_nodes,
                                     num_classes=self.num_classes)
        self.train_dataset_for_test=DataLoader(self.data_path,
                                      self.labels_path,
                                      self.edges_path,
                                      idx_path=self.idx_train,
                                      model_name=self.model_name,
                                      num_features= self.num_features,
                                      num_nodes=self.num_nodes,
                                      num_classes=self.num_classes,
                                      )
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, 
                                                   batch_size=self.batch_size, 
                                                   shuffle=True,
                                                   drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, 
                                                   batch_size=self.batch_size, 
                                                   shuffle=False,
                                                   sampler=torch.utils.data.SequentialSampler(self.test_dataset),
                                                   drop_last=False)
        self.train_loader_for_test = torch.utils.data.DataLoader(self.train_dataset, 
                                                   batch_size=self.batch_size, 
                                                   shuffle=False,
                                                   drop_last=False)
       
    def load_model(self):
        if self.model_name=="aagcn":
            self.model = aagcn_network( graph=self.edge_index ,
                                       num_person=1,
                                       num_nodes=self.num_nodes,
                                       num_subset=self.num_subset, 
                                       in_channels=self.num_features,
                                       drop_out=0.5, 
                                       adaptive=self.adaptive, 
                                       attention=self.attention,
                                       kernel_size=self.kernel_size,
                                       hidden_size=self.hidden_size,
                                       bn=self.bn,
                                       stride=self.strid
                                       )
        elif self.model_name=="a3tgcn":
            self.model=A3TGCN2_network(edge_index=self.edge_index,
                                       node_features=self.num_features,
                                       num_nodes=self.num_nodes,
                                       periods=self.TS,
                                       batch_size=self.batch_size)
        else:
            raise ValueError("No model with such name ", self.model_name )
        
        if(self.continue_training):
            self.load_pretraind(self.pretrain_model)
        self.model.to(self.device)
    def load_pretraind(self,path):
         self.model.load_state_dict(torch.load(path))
         print("Pre trained model is loaded...")

    def load_optimizer(self):
        if self.optimizer_name=="SGD":
            self.optimizer = torch.optim.SGD(list(self.model.parameters()),lr = self.lr,momentum = 0.9,weight_decay=0.0001)
        elif self.optimizer_name=="adam":
            self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr,weight_decay=0.0001)
        self.scheduler=lr_scheduler.StepLR(self.optimizer, self.step_decay,self.weight_decay)
        """
        for var_name in self.optimizer.state_dict():
            print(var_name, '\t', self.optimizer.state_dict()[var_name])
        """
    def load_loss(self):
        #self.loss=torch.nn.CrossEntropyLoss().to(self.device)
        self.loss=torch.nn.MSELoss().to(self.device)
        #self.loss=torch.nn.L1Loss().to(self.device)
    def apply_augmentain(self,snapshot):
        x_s,y_s=snapshot
        x_transormed=torch.zeros_like(x_s)
        for i in range( 0,len(x_s)):
            x,y=self.transform((x_s[i].cpu().numpy(),y_s[i]))
            x_transormed[i]=torch.tensor(x)
        return x_transormed,y_s


    def train(self):
        avg_loss = 0.0
        
        self.model.train()
        tq=tqdm(self.train_loader)
        for i,snapshot in enumerate(tq):
            x,label=snapshot
            x=x.to(self.device)
            label = label.to(self.device) 
            #forward
            y_hat = self.model(x)
            loss=self.loss(y_hat,label.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Clip gradients
            self.optimizer.step()
            self.optimizer.zero_grad()
            tq.set_description(f"train: Loss batch: {loss}")
            avg_loss += loss
        avg_loss = avg_loss / (i+1)    
        acc,cm,f1,_,_=self.calc_accuracy(mode="train")
        return avg_loss,acc,f1
    def get_min_max(self,x,min,max):
        m_found=np.min(x)
        ma_found=np.max(x)
        if  m_found< min:
            min=m_found
        if ma_found>max:
            max=ma_found
        return min,max

    def eval(self):
        self.model.eval()
        avg_loss = 0.0
        tq=tqdm(self.test_loader)
        with torch.no_grad():
            for i,snapshot in enumerate(tq):
                x,label=snapshot
                x=x.to(self.device)
                label = label.to(self.device) 
                y_hat = self.model(x) 
                loss=self.loss(y_hat,label.float())
                avg_loss += loss
                tq.set_description("Test Loss batch: {}".format(loss))
        avg_loss = avg_loss / (i+1)
        acc,cm,f1,_,_=self.calc_accuracy(mode="test")
        return avg_loss,acc,f1
    
    def calc_accuracy(self,path_model=None,mode="test",log=True):
        targests=[]
        predicted=[]
        if path_model!=None:
            self.load_pretraind(path_model)
        self.model.eval()
        count=0
        sample=0
        min_found=100
        max_found=0
        max_classes=np.max(self.classes)
        if mode=="test":
            tq=tqdm(self.test_loader)
        else:
            tq=tqdm(self.train_loader_for_test)
        with torch.no_grad():
            for i,snapshot in enumerate(tq):
                x,label=snapshot
                x=x.to(self.device)
                y_hat = self.model(x)
                y_hat=y_hat.cpu().numpy()
                min_found,max_found=self.get_min_max(y_hat,min_found,max_found)
                y_hat=[x * max_classes for x in y_hat]
                y_hat=np.round(y_hat).tolist()
                predicted.append(y_hat)
                label=label.tolist()
                label=[x * max_classes for x in label]
                targests.append(label)
                for k in range(0,len(label)):
                    sample=sample+1
                    if label[k] == y_hat[k]:
                        count=count+1
        
        targests=list(chain.from_iterable(targests))
        predicted=list(chain.from_iterable(predicted)) 
        
        cm=confusion_matrix( targests,predicted,labels=self.classes)
       
        f1=multiclass_f1_score(torch.tensor(predicted),torch.tensor(targests),num_classes=self.num_classes)
        #The precision is the ratio tp / (tp + fp)
        #The recall is the ratio tp / (tp + fn)
        accuracy_class=100*cm.diagonal()/cm.sum(1)
        acc=count/sample
        print("accuracy",acc)
        print("F1",f1)
        print("max value:",max_found,"min value:",min_found)
        print(cm,accuracy_class)
        p,r=self.get_precision_recall(cm)
       
        if log:
            wandb.log({f"conf_matrix_{mode}" : wandb.plot.confusion_matrix( 
                preds=predicted, y_true=targests,
                class_names=self.classes)})
            """
            wandb.log({f"pr_{mode}" : wandb.plot.pr_curve(targests,predicted,
                        labels=None, classes_to_plot=None)})
            """
            wandb.log({f"f1_{mode}":f1,f"precison_{mode}":p,f"recall_{mode}":r,f"{mode}_accuracy":acc})
            if mode=="test":
                self.cm.append(cm)
        return acc,cm,f1,accuracy_class,[max_found,min_found]
    def get_precision_recall(self,cm):
        
        true_pos = np.diag(cm)
        false_pos = np.sum(cm, axis=0) 
        false_neg = np.sum(cm, axis=1)
        precision=np.zeros(cm.shape[0])
        recall=np.zeros(cm.shape[0])
        for i in range(0,len(true_pos)):
            if false_pos[i]!=0:
                precision[i]=true_pos[i]/false_pos[i]
            if false_neg[i]!=0:
                recall[i]=true_pos[i]/false_neg[i]
        precision = np.mean(precision)
        recall = np.mean(recall)
        return precision,recall
    def get_embedding(self,path=None):
        self.model_embed= aagcn_network( graph=self.edge_index ,
                                        num_person=1,
                                        num_nodes=self.num_nodes,
                                        num_subset=self.num_subset, 
                                        in_channels=self.num_features,
                                        drop_out=0.5,
                                        adaptive=self.adaptive, 
                                        attention=self.attention,
                                        embed=True,
                                        kernel_size=self.kernel_size,
                                        bn=self.bn,
                                        stride=self.strid)
        if path==None:
            self.model_embed.load_state_dict(self.model.state_dict())
        else:
            self.model_embed.load_state_dict(torch.load(path))
        self.model_embed.to(self.device)
        self.model_embed.eval()
        class_embed_all=[]
        predicted_class_all=[]
        embed_all=[]
        tq=tqdm(self.test_loader)
        max_classes=np.max(self.classes)
        with torch.no_grad():
            for i,snapshot in enumerate(tq):
                x,label=snapshot
                initial_label=label
                x=x.to(self.device)
                y_hat,embed_vectors = self.model_embed(x)
                y_hat=y_hat.tolist()
                label=[x * max_classes for x in label]
                y_hat=[x * max_classes for x in y_hat]
                y_hat=np.round(y_hat).tolist()
                bs,t_step,dim_emb=embed_vectors.shape
                #try with y_hat instead of true predicted
                class_embed=np.repeat(label,t_step)
                predictes=np.repeat(y_hat,t_step)
                predicted_class_all.append(predictes)
                class_embed_all.append(class_embed)
                embed_vectors=embed_vectors.view(-1,dim_emb).cpu().numpy()
                embed_all.append(embed_vectors)
                assert class_embed.shape[0]==embed_vectors.shape[0]
            class_embed_all=np.concatenate(class_embed_all)
            embed_all=np.concatenate(embed_all)
            predicted_class_all=np.concatenate(predicted_class_all)
            print(class_embed_all)
        return embed_all,class_embed_all,predicted_class_all,initial_label
    def visualize_one_cm(self,cm,title="Confusion_matrix"):
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        ax.set_title(title)
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        image_wand = wandb.Image(image, caption=title)
        wandb.log({title: image_wand})
    def log_parameters(self):
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            str_par=""
            for i in self.config.keys():
                str_par+=i +" : "+ str(config[i])+"\n"
                
            f.write(" Parametrs:\n {}".format(str_par))
    def log_results(self,epoch,avg_train_loss,avg_test_loss,avg_test_acc,avg_train_acc):
        lr=self.optimizer.param_groups[0]['lr']
        result="Epoch {}, Train_loss: {} ,eval loss: {} ,eval_accuracy:{},train_accuracy{},lr:{} \n".format(epoch +1,avg_train_loss,avg_test_loss,avg_test_acc,avg_train_acc,lr)
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            f.write(result)
            print(result)
        if ((epoch+1)%5==0):
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "ckpt_%d.pkl" % (epoch +1)))
                print('Saved new checkpoint  ckpt_{epoch_}.pkl'.format(epoch_=epoch+1))

        
        wandb.log({"epoche": epoch, "train_loss":avg_train_loss,"test_loss":avg_test_loss,"lr":lr})

    def log_final_cm(self):

        avg_accuracy,cm,f1,accuracy_class,max_min= self.calc_accuracy(self.log_dir+"/best_model.pkl",log=False)
        
        self.visualize_one_cm(cm,title="best_model"+self.name)
        
        with open(self.log_dir+'/log.txt', 'a') as f:
                    f.write("results_best_model:\n Avg_Accuracy: {}\n max_min_value{}\n cm:{}\n class_accuracy{}".format(avg_accuracy,max_min,cm,accuracy_class))
        avg_accuracy,cm,f1,accuracy_class,max_min= self.calc_accuracy(self.log_dir+"/ckpt_{}.pkl".format(self.num_epoch),log=False)
        self.visualize_one_cm(cm,title="last_epoch"+self.name)
        
        with open(self.log_dir+'/log.txt', 'a') as f:
                    f.write("results_last_epoche:\n Avg_Accuracy: {}\n max_min_value{}\n cm:{}\n class_accuracy{}".format(avg_accuracy,max_min,cm,accuracy_class))
        
        avg_accuracy,cm,f1,accuracy_class,max_min= self.calc_accuracy(self.log_dir+"/ckpt_{}.pkl".format(self.num_epoch),mode="train",log=False)
        self.visualize_one_cm(cm,title="last_epoch_train"+ self.name)
        with open(self.log_dir+'/log.txt', 'a') as f:
                    f.write("results_last_epoch_on_train:\n Avg_Accuracy: {}\n max_min_value{}\n cm:{}\n class_accuracy{}".format(avg_accuracy,max_min,cm,accuracy_class))
    def save_best_model(self,prev_acc,acc):
        if acc > prev_acc:
            with open(self.log_dir+'/log.txt', 'a') as f:
                f.write("saved_a new best_model\n ")
            torch.save(self.model.state_dict(),self.log_dir+"/best_model.pkl")
            print('Best model in {dir}/best_model.pkl'.format(dir=self.log_dir))
            prev_acc=acc
        return prev_acc
         
    def run(self,name=None):
        wandb.init(project="New data with centroid velocity",config=self.config,name=name)
        wandb.run.log_code(".")
        wandb.watch(self.model,self.loss,log="all",log_freq=1,log_graph=True)
        self.set_log_dir(name)
        self.log_parameters()
        prev_best_acc = 0.0
        for epoch in range(self.num_epoch):
            avg_train_loss,avg_train_acc,f1_train=self.train()
            avg_test_loss,avg_test_acc,f1_test= self.eval()
            self.log_results(epoch,avg_train_loss,avg_test_loss,avg_test_acc,avg_train_acc)
            prev_best_acc=self.save_best_model(prev_best_acc,f1_test)
            self.scheduler.step()
        
        np.save(self.log_dir+"/cm.npy",self.cm) 
        self.log_final_cm()

        
if __name__=="__main__":
    torch.manual_seed(100)

    name_exp="open_face"
    parent_folder="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
    config_file=open(parent_folder+"/config/"+name_exp+".yml", 'r')
    config = yaml.safe_load(config_file)

    trainer=Trainer(config=config)
    trainer.run(config["log_name"])
    #trainer.calc_accuracy()
# %%

