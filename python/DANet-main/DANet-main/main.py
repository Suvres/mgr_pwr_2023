import os
os.system(f"pip install torch>=1.4.0 category_encoders yacs tensorboard>=2.2.2 qhoptim")

from DAN_Task import DANetClassifier, DANetRegressor
import argparse
import torch.distributed
import torch.backends.cudnn
from sklearn.metrics import accuracy_score, mean_squared_error
from data.dataset import get_data, quantile_transform
from lib.utils import normalize_reg_label
from qhoptim.pyt import QHAdam
from config.default import cfg
import pandas as pd
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_args():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.backends.cudnn.benchmark = True if len('1') < 2 else False

    from azureml.core import Run
    run = Run.get_context(allow_offline=True)
    # access to current workspace
    ws = run.experiment.workspace

    from azureml.core import Dataset
    dataset = Dataset.get_by_name(ws, name='config-danet')
    dataset.download(target_path='.', overwrite=True)


    cfg.merge_from_file('./cardio.yaml')
    cfg.freeze()
    task = cfg.task
    seed = cfg.seed
    train_config = {'dataset': 'cardio', 'resume_dir': cfg.resume_dir, 'logname': cfg.logname}
    fit_config = dict(cfg.fit)
    model_config = dict(cfg.model)
    print('Using config: ', cfg)

    return train_config, fit_config, model_config, task, seed, len('1')

def set_task_model(model_config, fit_config, seed=1):
    
    clf = DANetClassifier(
        optimizer_fn=QHAdam,
        optimizer_params=dict(lr=fit_config['lr'], weight_decay=1e-5, nus=(0.8, 1.0)),
        scheduler_params=dict(gamma=0.95, step_size=20),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        layer=model_config['layer'],
        base_outdim=model_config['base_outdim'],
        k=model_config['k'],
        drop_rate=model_config['drop_rate'],
        seed=seed
    )
    eval_metric = ['accuracy']

    return clf, eval_metric


def danet(dataframe1):
    print('===> Setting configuration ...')
    train_config, fit_config, model_config, task, seed, n_gpu = get_args()
    logname = None if train_config['logname'] == '' else train_config['dataset'] + '/' + train_config['logname']
    print('===> Getting data ...')

    dataframe1 = dataframe1.replace([np.inf, -np.inf], np.nan)
    dataframe1 = dataframe1.fillna(0)
    dataframe_c = dataframe1
    
    dataframe_test, dataframe1_valid = np.split(dataframe1, [int(.4*len(dataframe1))])
    
    X_train = dataframe_c.drop(["Label"], axis=1).values
    y_train = dataframe_c.loc[:, "Label"].values

    X_valid = dataframe1_valid.drop(["Label"], axis=1).values
    y_valid = dataframe1_valid.loc[:, "Label"].values

    X_test = dataframe_test.drop(["Label"], axis=1).values
    y_test = dataframe_test.loc[:, "Label"].values
   
    # X_train, X_valid, X_test = quantile_transform(X_train, X_valid, X_test)
    
    clf, eval_metric = set_task_model(model_config, fit_config, seed)
    
    clf.fit(    
        X_train=X_train, y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=['valid'],
        eval_metric=eval_metric,
        max_epochs=fit_config['max_epochs'], patience=fit_config['patience'],
        batch_size=fit_config['batch_size'], virtual_batch_size=fit_config['virtual_batch_size'],
        logname=logname,
        resume_dir=train_config['resume_dir'],
        n_gpu=n_gpu
    )
    
    preds_test = clf.predict(X_test)

    test_acc = accuracy_score(y_pred=preds_test, y_true=y_test)
    print(f"FINAL TEST ACCURACY FOR {train_config['dataset']} : {test_acc}")
