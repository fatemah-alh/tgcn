#%%
import torch

import numpy as np
from itertools import chain
from models.aagcn import aagcn_network
from models.a3tgcn import A3TGCN2_network
from dataloader import DataLoader
from tqdm import tqdm 
import torch.optim.lr_scheduler as lr_scheduler
import datetime
import os
import yaml
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('GTK3Cairo')

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
        self.output_features=config['output_features']
        self.TS=config['TS']
        self.continue_training=config['continue_training']
        self.pretrain_model=config['pretrain_model']
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
        self.set_log_dir()
        self.set_device()
        self.edge_index=torch.LongTensor(np.load(self.edges_path)).to(self.device)
        self.load_datasets()
        self.load_model()
        self.load_optimizer()
        self.load_loss()
        self.init_writer()

    def set_log_dir(self):
        self.date=datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.log_dir=os.path.join(self.LOG_DIR,self.date)
        os.makedirs(self.log_dir,exist_ok=True)
    def set_device(self):
        if torch.cuda.is_available():
            print("set cuda device")
            self.device="cuda"
            torch.cuda.set_device(self.gpu)
        else:
            self.device="cpu"
            print('Warning: Using CPU')
    def init_writer(self):
        writer_path=self.parent_folder+'writer/' + self.name_exp+self.date
        os.makedirs(writer_path,exist_ok=True)
        self.writer = SummaryWriter(writer_path)
    def load_datasets(self):
        self.train_dataset=DataLoader(self.data_path,self.labels_path,self.edges_path,idx_path=self.idx_train,model_name=self.model_name,num_features= self.num_features,num_nodes=self.num_nodes)
        self.test_dataset=DataLoader(self.data_path,self.labels_path,self.edges_path,idx_path=self.idx_test,model_name=self.model_name,num_features= self.num_features,num_nodes=self.num_nodes)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, 
                                                   batch_size=self.batch_size, 
                                                   shuffle=True,
                                                   drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, 
                                                   batch_size=self.batch_size, 
                                                   shuffle=False,
                                                   sampler=torch.utils.data.SequentialSampler(self.test_dataset),
                                                   drop_last=False)
    def load_model(self):
        if self.model_name=="aagcn":
            self.model = aagcn_network( graph=self.edge_index ,num_person=1,num_nodes=self.num_nodes,num_subset=self.num_subset, in_channels=self.num_features,drop_out=0.5, adaptive=self.adaptive, attention=True)
        elif self.model_name=="a3tgcn":
            self.model=A3TGCN2_network(edge_index=self.edge_index,node_features=self.num_features,num_nodes=self.num_nodes,periods=self.TS,batch_size=self.batch_size)
        else:
            raise ValueError("No model with such name ", self.model_name )
        """
        if(self.continue_training):
                path_pretrained_model=os.path.join(self.LOG_DIR,"{}/best_model.pkl".format(self.pretrain_model))
                self.model.load_state_dict(torch.load(path_pretrained_model))
                print("Pre trained model is loaded...")
        print(self.model.parameters)
        """
        self.model.to(self.device)

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
        self.MAE=torch.nn.L1Loss().to(self.device)
        
    def train(self):
        avg_loss = 0.0
        MAE=0.0
        self.model.train()
        tq=tqdm(self.train_loader)
        for i,snapshot in enumerate(tq):
            x,y=snapshot
            #Move tensors to device
            x=x.to(self.device)
            label = y.to(self.device) 
            #forward
            y_hat = self.model(x)
            loss=self.loss(y_hat,label.float())
            MAE_batch=self.MAE(y_hat,label.float())
            #calc gradient and backpropagation
            loss.backward()
            self.optimizer.step()
            #check gradient, if some tensors has been detached from CG
            for name, param in self.model.named_parameters():
               if param.grad==None:
                   raise RuntimeError(f"Gradient is None {name}" )
            self.optimizer.zero_grad()
            
            self.writer.add_scalars('train loss batch', {"loss":loss}, i)
            tq.set_description("train: Loss batch: {},MAE:{}".format(loss,MAE_batch))
            avg_loss += loss
            MAE += MAE_batch
        return avg_loss/(i+1),MAE/(i+1)

    def eval(self):
        self.model.eval()
        avg_loss = 0.0
        MAE=0.0
        tq=tqdm(self.test_loader)
        with torch.no_grad():
            for i,snapshot in enumerate(tq):
                x,y=snapshot
                x=x.to(self.device)
                label = y.to(self.device) 
                y_hat = self.model(x) 
                loss=self.loss(y_hat,label.float())
                MAE_batch=self.MAE(y_hat,label.float())
                avg_loss += loss
                self.writer.add_scalars('Eval loss batch', {"loss":loss}, i)
                tq.set_description("Test Loss batch: {}".format(loss))
                MAE += MAE_batch
        avg_loss = avg_loss / (i+1)
        MAE=MAE/(i+1)
        return avg_loss,MAE
    
    def calc_accuracy(self,path_model=None):
        targests=[]
        predicted=[]
        test_dataset=DataLoader(self.data_path,self.labels_path,self.edges_path,normalize_labels=False,idx_path=self.idx_test,model_name=self.model_name,num_features= self.num_features,num_nodes=self.num_nodes)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=self.batch_size, 
                                                  shuffle=False,
                                                  sampler=torch.utils.data.SequentialSampler(test_dataset),
        
                                                  drop_last=False)
        if path_model!=None:
            self.model.load_state_dict(torch.load(path_model))
            print("Pre trained model is loaded...")
        elif self.pretrain_model!="None":
            path_pretrained_model=self.LOG_DIR+"{}/best_model.pkl".format(self.pretrain_model)
            self.model.load_state_dict(torch.load(path_pretrained_model))
            print("Pre trained model is loaded...")
        self.model.eval()
        count=0
        sample=0
        min=100
        max=0
        tq=tqdm(test_loader)
        with torch.no_grad():
            for i,snapshot in enumerate(tq):
                x,y=snapshot
                x=x.to(self.device)
                label = y.to(self.device)
               
                targests.append(y.tolist())
                
                y_hat = self.model(x).tolist()
                
                min_found=np.min(y_hat)
                max_found=np.max(y_hat)
                if  min_found< min:
                    min=min_found
                if max_found>max:
                    max=max_found
                y_hat=[x * 4 for x in y_hat]
                
                y_hat=np.round(y_hat)
                predicted.append(y_hat.tolist())
                for k in range(0,len(label)):
                    sample=sample+1
                    if label[k] == y_hat[k]:
                        count=count+1
        print("accuracy",count/sample)
        print("max value:",max,"min value:",min)
        
        targests=list(chain.from_iterable(targests))#np.array(targests).flatten()
        predicted=list(chain.from_iterable(predicted)) #np.array(predicted).flatten()
       
        cm=confusion_matrix( targests,predicted,labels=[0,1,2,3,4])
        print(cm)
        self.cm.append(cm)
        accuracy_class=100*cm.diagonal()/cm.sum(1)
        print(accuracy_class)
        
        """
        sns.heatmap(cm,annot=True)
        plt.ylabel('Prediction',fontsize=13)
        plt.xlabel('Actual',fontsize=13)
        plt.title('Confusion Matrix',fontsize=17)
        plt.show()
        """
        return count/sample,cm,accuracy_class,[max,min]
    def run(self):
        previous_best_avg_test_acc = 0.0
        #previous_best_avg_loss=1000000
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            str_par=""
            for i in config.keys():
                str_par+=i +" : "+ str(config[i])+"\n"
                
            f.write(" Parametrs:\n {}".format(str_par))
            
        for epoch in range(self.num_epoch):
            avg_train_loss,MAE_train=self.train()
        
            if ((epoch+1)%5==0):
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "ckpt_%d.pkl" % (epoch + 1)))
                print('Saved new checkpoint  ckpt_{epoch}.pkl , avr loss {l}'.format( epoch=epoch+1, l=avg_train_loss))
          
            avg_test_loss,MAE_test= self.eval()
            avg_accuracy,_,_,_=self.calc_accuracy()
            self.writer.add_scalars("Loss training and evaluating",{'train_loss': avg_train_loss,'eval_loss': avg_test_loss,}, epoch)

            result="Epoch {}, Train_loss: {} ,eval loss: {} ,eval_accuracy:{},lr:{} \n".format(epoch + 1,avg_train_loss,avg_test_loss,avg_accuracy,self.optimizer.param_groups[0]['lr'])
            with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
                f.write(result)
            print(result)
            

        
            if avg_accuracy > previous_best_avg_test_acc:
                with open(self.log_dir+'/log.txt', 'a') as f:
                    f.write("saved_a new best_model\n ")
                torch.save(self.model.state_dict(),self.log_dir+"/best_model.pkl")
                print('Best model in {dir}/best_model.pkl'.format(dir=self.log_dir))
                #previous_best_avg_test_acc = avg_test_acc
                previous_best_avg_test_acc=avg_accuracy
            self.scheduler.step()
        
        np.save(self.log_dir+"/cm.npy",self.cm)    
        avg_accuracy,cm,accuracy_class,max_min= self.calc_accuracy(self.log_dir+"/best_model.pkl")
        with open(self.log_dir+'/log.txt', 'a') as f:
                    f.write("results_best_model:\n Avg_Accuracy: {}\n max_min_value{}\n cm:{}\n class_accuracy".format(avg_accuracy,max_min,cm,accuracy_class))
        avg_accuracy,cm,accuracy_class,max_min= self.calc_accuracy(self.log_dir+"/ckpt_{}.pkl".format(self.num_epoch))
        with open(self.log_dir+'/log.txt', 'a') as f:
                    f.write("results_best_model:\n Avg_Accuracy: {}\n max_min_value{}\n cm:{}\n class_accuracy".format(avg_accuracy,max_min,cm,accuracy_class))
               

if __name__=="__main__":
    torch.manual_seed(100)

    name_exp="open_face"
    #name_exp = 'mediapipe'
    #name_exp = 'dlib'
    #name_exp = 'minidata'
    parent_folder="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
    config_file=open(parent_folder+"/config/"+name_exp+".yml", 'r')
    config = yaml.safe_load(config_file)

    trainer=Trainer(config=config)
    trainer.run()
    
    #trainer.calc_accuracy()
# %%

