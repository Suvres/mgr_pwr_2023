import os
os.system(f"pip install torch>=1.4.0 category_encoders yacs tensorboard>=2.2.2 qhoptim")

from DAN_Task import DANetClassifier, DANetRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from lib.multiclass_utils import infer_output_dim
from lib.utils import normalize_reg_label
import numpy as np
from data.dataset import get_data
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_args():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    dataset = 'cardio'
    model_file = 'weight/best.pth'
    task = 'classification'

    return dataset, model_file, task, len('1')

def set_task_model(task):
    clf = DANetClassifier()
    return clf

def prepare_data(task, y_train, y_valid, y_test):
    output_dim = 1
    mu, std = None, None
    output_dim, train_labels = infer_output_dim(y_train)
    
    target_mapper = {class_label: index for index, class_label in enumerate(train_labels)}
    
    y_train = np.vectorize(target_mapper.get)(y_train)
    y_valid = np.vectorize(target_mapper.get)(y_valid)
    y_test = np.vectorize(target_mapper.get)(y_test)

    return output_dim, std, y_train, y_valid, y_test

def predict(dataframe1, dataframe2):
    dataset, model_file, task, n_gpu = get_args()
    print('===> Getting data ...')
    
    dataframe1 = dataframe1.replace([np.inf, -np.inf], np.nan)
    dataframe1 = dataframe1.fillna(0)

    dataframe2 = dataframe2.replace([np.inf, -np.inf], np.nan)
    dataframe2 = dataframe2.fillna(0)

    dataframe1, dataframe1_valid = np.split(dataframe1, [int(.1*len(dataframe1))])

    y_train = dataframe1.loc[:, "Label"].values
    y_valid = dataframe1_valid.loc[:, "Label"].values

    X_test = dataframe2.drop(["Label"], axis=1).values
    y_test = dataframe2.loc[:, "Label"].values
    
    
    output_dim, std, y_train, y_valid, y_test = prepare_data(task, y_train, y_valid, y_test)
    clf = set_task_model(task)
    
    filepath = model_file
    from azureml.core import Run
    run = Run.get_context(allow_offline=True)
    # access to current workspace
    ws = run.experiment.workspace

    from azureml.core import Dataset
    datastore = ws.get_default_datastore()    
        
    target_path = 'UI/tmp/best.pth'

    dataset = Dataset.File.from_files(path=(datastore, target_path))
    dataset.download(target_path='.', overwrite=True)
    
    clf.load_model('./best.pth', input_dim=X_test.shape[1], output_dim=output_dim, n_gpu=n_gpu)

    preds_test = clf.predict(X_test)
    preds_prob = clf.predict_proba(X_test)[:, 1]

    print("< acc ------------------------------------>")
    acc = accuracy_score(y_pred=preds_test, y_true=y_test)
    print("< prec ------------------------>")

    prec = precision_score(y_test, preds_test)
    print("< rec ------------------------>")
    
    rec = recall_score(y_test, preds_test)
    print("< f1------------------------>")

    f1 = 2 * prec * rec / (rec + prec)
    print("< auc ------------------------>")
    
    auc = roc_auc_score(y_test, preds_prob)
    print("<------------------------>")
    return pd.DataFrame({
        'model': ['DANET'],
        'accuracy': [acc],
        'precision': [prec],
        'recall': [rec],
        'f1': [f1],
        'auc': [auc]
    })