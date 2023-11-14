import torch
import numpy as np 
import wandb
import os
from helper.Config import Config
from sklearn.metrics import  ConfusionMatrixDisplay
from PIL import Image
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, config:Config,model=None,loss=None) -> None:
        
        self.config=config
        self.log_dir = self._prepare_log_dir()
        self.log_parameters()
        self._init_wandb(config.project_name,model,loss)
    def _prepare_log_dir(self):
        log_dir = os.path.join(self.config.parent_folder, self.config.LOG_DIR, self.config.log_name)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def log_metrics(self, metrics: dict, step: int):
        wandb.log(metrics, step=step)

    def save_checkpoint(self, model, optimizer, epoch):
        checkpoint_path = os.path.join(self.log_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, checkpoint_path)

    def log_message(self, message: str):
        
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as log_file:
            log_file.write(message + "\n")
        
    # ... other logging methods such as for visualizations
    def log_max_min(self,values,mode):
        msg=f"Mode: {mode}, Max value :{np.max(values)}, Min value:{np.min(values)} "
        print(msg)
        return msg
       
    def log_parameters(self):
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            str_par=""
            for variable_name, variable_type in Config.__annotations__.items():
                value = getattr(self.config, variable_name)
                str_par+=f"{variable_name}: {value} \n"
            
            f.write(f" Parametrs:\n {str_par}")
    def _init_wandb(self,proj_name,model,loss):
        wandb.init(project=proj_name,config=self.config,name=self.config.log_name)
        wandb.watch(model,loss,log="all",log_freq=1,log_graph=True)

    def log_epoch(self,epoch,mode,mse_err,rmse_err,mea_err,acc,f1_macro,f1_micro,p,r,lr):
        
        result=f"Epoch: {epoch}, {mode}_acc:{acc},{mode}_f1_macro:{f1_macro},{mode}_f1_micro:{f1_micro},{mode}_p:{p},{mode}_r:{r}, {mode}_MSE: {mse_err} ,{mode}_RMSE: {rmse_err} ,{mode}_MEA:{mea_err},lr:{lr} \n"
        print(result)
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            f.write(result)
            
    def log_epoch_wandb(self,epoch,mode,mse_err,rmse_err,mea_err,acc,f1_macro,f1_micro,p,r):
       
        wandb.log({ f"{mode}_loss": mse_err, f"{mode}_acc": acc, f"{mode}_f1":f1_micro,f"{mode}_precison":p,f"{mode}_recall":r})

    def save_best_model(self,model):
        with open(self.log_dir+'/log.txt', 'a') as f:
            f.write("Saved a new best_model\n ")
        torch.save(model,self.log_dir+"/best_model.pkl")
        print('Best model in {dir}/best_model.pkl'.format(dir=self.log_dir))

    def log_cm_colored_wandb(self,cm,title="Confusion_matrix"):
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        ax.set_title(title)
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        image_wand = wandb.Image(image, caption=title)
        wandb.log({title: image_wand})
    def round_values(self,values,normalized_labels,max_classes): 
        #Repeated function also in evaluation.
        if normalized_labels:
            values=[x * max_classes  for x in values]
        values=np.round(values)
        values=torch.clamp(torch.tensor(values), min=0, max=max_classes)
        
        return values
    def log_cm_wandb(self,mode,targets,predicted,classes):
        predicted= self.round_values(predicted,self.config.normalize_labels,np.max(classes))
        targets=self.round_values(targets,self.config.normalize_labels,np.max(classes))
        print(predicted,targets)
        wandb.log({f"conf_matrix_{mode}" : wandb.plot.confusion_matrix( 
                preds=predicted, y_true=targets,
                class_names=classes)})
    def save_results(self,i,acc,loss,path):
        with open(path+".txt", 'a') as f:
                f.write(f"Subject: {i}, acc: {acc}, loss: {loss}")
  
         