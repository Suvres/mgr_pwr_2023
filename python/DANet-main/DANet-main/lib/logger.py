import os
import torch
from datetime import datetime, timezone
from torch.utils.tensorboard import SummaryWriter

class Train_Log():
    def __init__(self, logname, resume_dir=None):
        time_str = datetime.now().strftime("%m-%d_%H%M")
        if resume_dir:
            self.resume_dir = os.path.join('./logs', resume_dir)
            self.log_dir = self.resume_dir

        else:
            self.log_dir = os.path.join('./logs/',  logname + '_' +time_str)

        self.writer = SummaryWriter(self.log_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def save_best_model(self, model):
        print('Save Best model!!')

        absolute=os.path.abspath('train')
        path= os.path.join(absolute, "best.pth")
        
        if not os.path.exists(absolute):
            os.mkdir(absolute)
        
        torch.save(model, path)
        
        from azureml.core import Run, Datastore
        from azureml.core import Dataset
        
        run = Run.get_context(allow_offline=True)
        # access to current workspace
        ws = run.experiment.workspace
        datastore: Datastore = ws.get_default_datastore()    
        
        target_path = 'UI/' + 'tmp'

        Dataset.File.upload_directory(absolute, (datastore, target_path), overwrite=True)
        print("<=== log save ===>")

    def save_log(self, log):
        mode = 'a' if os.path.exists(self.log_dir + '/log.txt') else 'w'
        logFile = open(self.log_dir + '/log.txt', mode)
        logFile.write(log + '\n')
        logFile.close()


    def save_tensorboard(self, info, epoch):
        for tag, value in info.items():
            self.writer.add_scalar(tag, value, global_step=epoch)
