#%%
import torch
import wandb
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

#wandb.login() #just  for first run
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
        self.MAE=torch.nn.L1Loss().to(self.device)
        
    def train(self):
        avg_loss = 0.0
        MAE=0.0
        self.model.train()
        tq=tqdm(self.train_loader)
        for i,snapshot in enumerate(tq):
            x,label=snapshot
            #Move tensors to device
            x=x.to(self.device)
            label = label.to(self.device) 
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
        MAE=0.0
        tq=tqdm(self.test_loader)
        with torch.no_grad():
            for i,snapshot in enumerate(tq):
                x,label=snapshot
                x=x.to(self.device)
                label = label.to(self.device) 
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
        if path_model!=None:
            self.load_pretraind(path_model)
        self.model.eval()
        count=0
        sample=0
        min_found=100
        max_found=0
        tq=tqdm(self.test_loader)
        with torch.no_grad():
            for i,snapshot in enumerate(tq):
                x,label=snapshot
                x=x.to(self.device)
                
                y_hat = self.model(x)
                y_hat=y_hat.cpu().numpy()
                min_found,max_found=self.get_min_max(y_hat,min_found,max_found)
                y_hat=[x * 4 for x in y_hat]
                y_hat=np.round(y_hat).tolist()
                predicted.append(y_hat)
                label=[x * 4 for x in label]
                targests.append(label)
                for k in range(0,len(label)):
                    sample=sample+1
                    if label[k] == y_hat[k]:
                        count=count+1
        
        targests=list(chain.from_iterable(targests))#np.array(targests).flatten()
        predicted=list(chain.from_iterable(predicted)) #np.array(predicted).flatten()
        cm=confusion_matrix( targests,predicted,labels=[0,1,2,3,4])
        self.cm.append(cm)
        accuracy_class=100*cm.diagonal()/cm.sum(1)

        print("accuracy",count/sample)
        print("max value:",max_found,"min value:",min_found)
        print(cm,accuracy_class)
        return count/sample,cm,accuracy_class,[max_found,min_found]
    def get_embedding(self,path=None):
        self.model_embed= aagcn_network( graph=self.edge_index ,num_person=1,num_nodes=self.num_nodes,num_subset=self.num_subset, in_channels=self.num_features,drop_out=0.5, adaptive=self.adaptive, attention=True,embed=True)
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
        with torch.no_grad():
            for i,snapshot in enumerate(tq):
                x,label=snapshot
                x=x.to(self.device)
                y_hat,embed_vectors = self.model_embed(x)
                y_hat=y_hat.tolist()
                label=[x * 4 for x in label]
                y_hat=[x * 4 for x in y_hat]
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
        return embed_all,class_embed_all,predicted_class_all

    def run(self):
        wandb.init(project="First_Run")
        wandb.config = self.config
        wandb.watch(self.model,self.loss,log="all",log_freq=10)
       

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
            wandb.log({"epoche": epoch, "train_loss":avg_train_loss,"test_loss":avg_test_loss,"test_accuracy":avg_accuracy})

        
            if avg_accuracy > previous_best_avg_test_acc:
                with open(self.log_dir+'/log.txt', 'a') as f:
                    f.write("saved_a new best_model\n ")
                torch.save(self.model.state_dict(),self.log_dir+"/best_model.pkl")
                print('Best model in {dir}/best_model.pkl'.format(dir=self.log_dir))
                previous_best_avg_test_acc=avg_accuracy
            self.scheduler.step()
        
        np.save(self.log_dir+"/cm.npy",self.cm)    
        avg_accuracy,cm,accuracy_class,max_min= self.calc_accuracy(self.log_dir+"/best_model.pkl")
        wandb.log({"cm_matrix_best_model":cm,"class_accuracy_best_model":accuracy_class,"Max_min_best_model":max_min})
        with open(self.log_dir+'/log.txt', 'a') as f:
                    f.write("results_best_model:\n Avg_Accuracy: {}\n max_min_value{}\n cm:{}\n class_accuracy{}".format(avg_accuracy,max_min,cm,accuracy_class))
        avg_accuracy,cm,accuracy_class,max_min= self.calc_accuracy(self.log_dir+"/ckpt_{}.pkl".format(self.num_epoch))
        with open(self.log_dir+'/log.txt', 'a') as f:
                    f.write("results_last_epoche:\n Avg_Accuracy: {}\n max_min_value{}\n cm:{}\n class_accuracy{}".format(avg_accuracy,max_min,cm,accuracy_class))
        wandb.log({"cm_matrix_last_epoche":cm,"class_accuracy_last_epoche":accuracy_class,"Max_min_last_epoch":max_min})
        self.model.to_onnx()
        wandb.save("model.onnx")

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

