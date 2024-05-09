import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np

def simple_gnb(dataset: pd.DataFrame, dataset_test: pd.DataFrame):
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.fillna(0)

    dataset_test = dataset_test.replace([np.inf, -np.inf], np.nan)
    dataset_test = dataset_test.fillna(0)

    x = dataset.drop(["Label"], axis=1)
    y = dataset.loc[:, "Label"].values

    x_test = dataset_test.drop(["Label"], axis=1)
    y_test = dataset_test.loc[:, "Label"].values

    x.replace([np.inf, -np.inf], np.nan)
    x_test.replace([np.inf, -np.inf], np.nan)

    x = x.fillna(0)
    x_test = x_test.fillna(0)

    nb = GaussianNB()
    nb.fit(x, y)

    y_pred = nb.predict(x_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = 2 * prec * rec / (rec + prec)
    auc = roc_auc_score(y_test, y_pred)
    
    data = pd.DataFrame({
        'model': ['Gaussian Naive Base - with GA'],
        'accuracy': [acc],
        'precision': [prec],
        'recall': [rec],
        'f1': [f1],
        'auc': [auc]
    })

    print(data)
    return data
