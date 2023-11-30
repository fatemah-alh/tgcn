#%%
import torch
import numpy as np
import random
from itertools import chain
from models.aagcn import aagcn_network
from models.a3tgcn import A3TGCN2_network
from tqdm import tqdm 
import torch.optim.lr_scheduler as lr_scheduler
#wandb.login() #just  for first run
from helper.Config import Config
from helper.DataHandler import DataHandler
from helper.Logger import Logger
from helper. Evaluation import Evaluation

class Trainer():
    def __init__(self, config) -> None:
        self.config= config
        self.classes=self.set_classes(self.config.num_classes)
        self.max_classes=np.max(self.classes)
        self.set_device()
        self.load_edges()
        if self.config.protocol=="hold_out":
            self.init_trainer()
    
    def set_classes(self,num_classes):
        classes_factory={2:[0,1],
                         3:[0,1,2],
                         5:[0,1,2,3,4]}
        if num_classes in classes_factory:
            return classes_factory[num_classes]
        else:
            raise(ValueError(f"No classes associated to the {num_classes} are found."))
    def init_trainer(self):

        self.datahandler=DataHandler(self.config)
        self.load_model()
        self.load_optimizer()
        self.load_loss()
        self.load_eval()
    def set_device(self):

        if torch.cuda.is_available():
            print("set cuda device")
            self.device="cuda"
            torch.cuda.set_device(self.config.gpu)
        else:
            self.device="cpu"
            print('Warning: Using CPU')

    def load_edges(self):
        self.edge_index=torch.LongTensor(np.load(self.config.parent_folder+self.config.edges_path)).to(self.device)
    
    def load_model(self):
        model_factory = {
            "aagcn": lambda: aagcn_network(
                graph=self.edge_index,
                num_person=1,
                num_nodes=self.config.n_joints,
                num_subset=self.config.num_subset,
                in_channels=self.config.num_features,
                drop_out=self.config.drop_out,
                adaptive=self.config.adaptive,
                attention=self.config.attention,
                kernel_size=self.config.t_kernel_size,
                hidden_size=self.config.hidden_size,
                bn=self.config.bn,
                stride=self.config.strid,
                
            ),
            "a3tgcn": lambda: A3TGCN2_network(
                edge_index=self.edge_index,
                node_features=self.config.num_features,
                num_nodes=self.config.n_joints,
                periods=self.config.TS,
                batch_size=self.config.batch_size
            )
        }

        if self.config.model_name in model_factory:
            self.model = model_factory[self.config.model_name]()
        else:
            raise ValueError(f"No model with the name {self.config.model_name} found.")

        if self.config.continue_training:
            self.load_pretrained(self.config.pretrain_model)
        self.model.to(self.device)

    def load_pretraind(self,path):
         self.model.load_state_dict(torch.load(path))
         print("Pre trained model is loaded...")

    def load_optimizer(self):
        opt_factory={ "SGD": lambda: torch.optim.SGD(list(self.model.parameters()),lr = self.config.lr,momentum = self.config.momentum,weight_decay=self.config.L2),
                      "adam": lambda:  torch.optim.Adam(list(self.model.parameters()), lr=self.config.lr,weight_decay=self.config.L2)
                      }
        if self.config.optimizer_name in opt_factory:
            self.optimizer = opt_factory[self.config.optimizer_name]()
        else:
            raise ValueError(f"No optimizer with the name {self.config.optimizer_name} found.")

        self.scheduler=lr_scheduler.StepLR(self.optimizer, self.config.step_decay, self.config.weight_decay)
     
    def load_loss(self):
        self.loss=torch.nn.MSELoss().to(self.device)
    def load_eval(self):
        self.evaluation=Evaluation(self.config)   
    def conc_aug_batch(self,x,y):
        b,l,c,t,n,m=x.size()
        x=x.view(-1 ,c, t, n, m)
        y=y.view(-1)
        idx=list(range(b*l))
        random.shuffle(idx)
        x=x[idx]
        y=y[idx]
        return x,y
    def set_log_dir(self):
        self.logger=Logger(self.config,model=self.model,loss=self.loss)

    def train(self):
        self.model.train()
        tq=tqdm(self.datahandler.train_loader)
        for i,snapshot in enumerate(tq):
            x,label=snapshot
            x=x.to(self.device)
            label = label.to(self.device)
            if self.config.concatenate and self.config.augmentaion:
                x,label=self.conc_aug_batch(x,label)
            y_hat = self.model(x)
            loss=self.loss(y_hat,label.float())
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)  # Clip gradients
            self.optimizer.step()
            self.optimizer.zero_grad()
            #y_hat=torch.clamp(y_hat, min=0, max=self.max_classes)
            tq.set_description(f"train loss batch {i}: {loss}")
    
  
    def eval(self,mode="test"):
        """
        Eval for training and test set. 
        """
        tq_factory={"train":self.datahandler.train_loader_for_test,
                    "test":self.datahandler.test_loader}
        targets=[]
        unrounded_predicted=[]
        
        tq=tqdm(tq_factory[mode])
        self.model.eval()
        with torch.no_grad():
            for i,snapshot in enumerate(tq):
                x,label=snapshot
                x=x.to(self.device)
                #label = label.to(self.device) 
                y_hat = self.model(x) 
               
                loss=self.loss(y_hat,label.float())
                targets.append(label.cpu().numpy())
                unrounded_predicted.append(y_hat.cpu().numpy())
                tq.set_description(f"{mode} loss batch {i} : {loss}")
        targets=list(chain.from_iterable(targets))# flatten is used with arrays not with lists
        unrounded_predicted=list(chain.from_iterable(unrounded_predicted)) 
        
        return targets,unrounded_predicted
    
    def get_results(self,mode,targets,unrounded_predicted,epoch,title=None,log_finale=False):
        if title:
            self.logger.log_message(title)
            print(title)
        
        acc,f1_macro,f1_micro,p,r,cm,acc_class= self.evaluation.calc_acc(targets,unrounded_predicted,self.classes,normalized_labels=self.config.normalize_labels) #self.calc_accuracy(mode="train")
        mse_err,rmse_err,mea_err= self.evaluation.calc_errors(targets,unrounded_predicted)
        #Start logg
        self.logger.log_epoch(epoch,mode,mse_err,rmse_err,mea_err,acc,f1_macro,f1_micro,p,r,self.optimizer.param_groups[0]['lr'])
        msg_min_max=self.logger.log_max_min(unrounded_predicted,mode)
        if log_finale:
            self.logger.log_message(msg_min_max)
            self.logger.log_cm_colored_wandb(cm,title=title)
            self.logger.log_message("cm: {}".format(cm))
            self.logger.log_message("acc_class:{}".format(acc_class))
            
        else:
            self.logger.log_epoch_wandb(epoch,mode,mse_err,rmse_err,mea_err,acc,f1_macro,f1_micro,p,r)
            print(cm)
            #self.logger.log_cm_wandb(mode=mode,targets=targets,predicted=unrounded_predicted,classes=self.classes)
        return f1_micro,rmse_err
    def get_embedding(self,path=None):
        self.model_embed= aagcn_network(graph=self.edge_index ,
                                        num_person=1,
                                        num_nodes=self.config.n_joints,
                                        num_subset=self.config.num_subset, 
                                        in_channels=self.config.num_features,
                                        drop_out=self.config.drop_out,
                                        adaptive=self.config.adaptive, 
                                        attention=self.config.attention,
                                        kernel_size=self.config.t_kernel_size,
                                        hidden_size=self.config.hidden_size,
                                        bn=self.config.bn,
                                        stride=self.config.strid,
                                        embed=True)
        
        if path==None:
            self.model_embed.load_state_dict(self.model.state_dict())
        else:
            self.model_embed.load_state_dict(torch.load(path))
        self.model_embed.to(self.device)
        self.model_embed.eval()
        class_embed_all=[]
        predicted_class_all=[]
        embed_all=[]
        
        tq=tqdm(self.datahandler.test_loader)
        with torch.no_grad():
            for i,snapshot in enumerate(tq):
                x,label=snapshot
                initial_label=label
                x=x.to(self.device)
                y_hat,embed_vectors = self.model_embed(x)
                y_hat=y_hat.tolist()
                if self.config.normalized_label:
                    label=[x * self.max_classes for x in label]
                    y_hat=[x * self.max_classes for x in y_hat]
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
    def get_all_outputs(self,path):
        self.model_embed= aagcn_network( graph=self.edge_index ,
                                        num_person=1,
                                        num_nodes=self.config.n_joints,
                                        num_subset=self.config.num_subset, 
                                        in_channels=self.config.num_features,
                                        drop_out=self.config.drop_out,
                                        adaptive=self.config.adaptive, 
                                        attention=self.config.attention,
                                        kernel_size=self.config.t_kernel_size,
                                        hidden_size=self.config.hidden_size,
                                        bn=self.config.bn,
                                        stride=self.config.strid,
                                        return_all_outputs=True
                                        )
        if path==None:
            self.model_embed.load_state_dict(self.model.state_dict())
        else:
            self.model_embed.load_state_dict(torch.load(path))
        self.model_embed.to(self.device)
        self.model_embed.eval()
        targets=[]
        unrounded_predicted=[]
        all_outputs=[]
        tq=tqdm(self.datahandler.test_loader)
        with torch.no_grad():
            for i,snapshot in enumerate(tq):
                x,label=snapshot
                x=x.to(self.device)
                y_hat,outputs = self.model_embed(x)
                targets.append(label.cpu().numpy())
                unrounded_predicted.append(y_hat.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
                
        targets=torch.tensor(list(chain.from_iterable(targets)))# flatten is used with arrays not with lists
        unrounded_predicted=torch.tensor(list(chain.from_iterable(unrounded_predicted)))
        all_outputs=torch.tensor(list(chain.from_iterable(all_outputs)))
    
        print(all_outputs.shape)
        targets=self.evaluation.round_values(targets,self.config.normalize_labels,self.max_classes)
        predicted=self.evaluation.round_values(unrounded_predicted,self.config.normalize_labels,self.max_classes)
        
        print(len(targets),len(unrounded_predicted),len(all_outputs)) 
        features=self.datahandler.test_dataset.__getattribute__("features")
        return targets,predicted,all_outputs,features

    def run(self):
        
        self.set_log_dir()
        best_acc = 0.0
        best_loss=0.0
        for epoch in range(self.config.num_epoch):

            self.train()
            targets,unrounded_predicted= self.eval(mode="test")
            f1_test,rmse_err_test=self.get_results("test",targets,unrounded_predicted,epoch=epoch)

            targets,unrounded_predicted= self.eval(mode="train")
            f1_train,rmse_err_train=self.get_results("train",targets,unrounded_predicted,epoch=epoch)
            if f1_test > best_acc:
                self.logger.save_best_model(self.model.state_dict())
                best_acc=f1_test
                best_loss=rmse_err_test
            self.scheduler.step()
        #self.log_dir+"/best_model.pkl" log the best model
        
        self.final_eval()
        return best_acc,best_loss
    def final_eval(self):
        self.logger.log_message("___________Training is Finished__________")
        targets,unrounded_predicted= self.eval(mode="test")
        f1_test,rmse_err_test=self.get_results("test",targets,unrounded_predicted,epoch="Final",title="Test_Last_Epoch",log_finale=True)

        targets,unrounded_predicted= self.eval(mode="train")
        f1_train,rmse_err_train=self.get_results("train",targets,unrounded_predicted,epoch="Final",title="Train_Last_Epoch",log_finale=True)

        self.load_pretraind(self.logger.log_dir+"/best_model.pkl")
        targets,unrounded_predicted= self.eval(mode="test")
        f1_test,rmse_err_test=self.get_results("test",targets,unrounded_predicted,epoch="Final",title="Best_Model",log_finale=True)

    def run_loso(self,type_="LE87",class_="binary",start=0,end=87): # or "LE67", "multi"

        folder_name=f"log/loso_{type_}_new/"
        log_loso=self.config.parent_folder+folder_name+f"log_loso_{type_}_{class_}"

        
        for i in range(start,end):
            idx_train=self.config.parent_folder+f"data/PartA/loso_{type_}/{i}/idx_train.npy"
            idx_test=self.config.parent_folder+f"data/PartA/loso_{type_}/{i}/idx_test.npy"
            self.LOG_DIR= self.config.parent_folder+folder_name+f"{i}/"
            self.log_name= f"1s+15k+{class_}+loso_{type_}_test{i}"
            
            self.config.idx_train=idx_train
            self.config.idx_test=idx_test
            self.config.LOG_DIR=self.LOG_DIR
            self.config.log_name=self.log_name
            
            self.init_trainer()
            acc,loss=self.run()
            self.logger.save_results(i,acc,loss,log_loso)

if __name__=="__main__":
    torch.manual_seed(100)

    config_file="open_face"
    parent_folder="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
    config =Config.load_from_file(parent_folder+"/config/"+config_file+".yml")

    trainer=Trainer(config=config)
    if config.protocol=="hold_out":
        trainer.run()
    else:
        #trainer.run_loso(type_="LE67",class_="binary",start=4)
        #trainer.run_loso(type_="LE67",class_="multi",start=0,end=5)
        #trainer.run_loso(type_="ME87",class_="multi",start=32,end=40)
        trainer.run_loso(type_="ME87",class_="binary",start=75,end=87)


# %%
