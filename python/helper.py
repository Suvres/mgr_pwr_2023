from typing import List

import pandas as pd
import numpy as np
from datetime import datetime
from azureml.core import Dataset, Run, Datastore

def result_log(result: pd.DataFrame, columns):
    dt = datetime.now()
    date = dt.strftime('%Y-%m-%d %H:%M:%S')
    columns_names = ",".join(columns)

    log = "[{0}] columns: [{1}], " \
          "accuracy: {2:.4f}, " \
          "precision: {3:.4f}, " \
          "recall: {4:.4f}, " \
          "auc: {5:.4f}, " \
          "f1: {5:.4f}\n".format(date, columns_names, result.loc[0, "accuracy"],
                                    result.loc[0, "precision"], result.loc[0, "recall"], result.loc[0, "auc"],
                                    result.loc[0, "f1"])

    # file = open('ga.log', 'a')
    # file.write(log)
    # file.close()


def read_b(x):
    if x < 1024:
        return "{0:.2f} B".format(x)
    elif x < (1024 * 1024):
        return "{0:.2f} KB".format(x/1024)
    elif x < (1024 * 1024 * 1024):
        return "{0:.2f} MB".format(x / (1024 * 1024))
    elif x < (1024 * 1024 * 1024 * 1024):
        return "{0:.2f} GB".format(x / (1024 * 1024 * 1024))


def save_solution(columns, genome):
    run = Run.get_context(allow_offline=True)
    ws = run.experiment.workspace

    datastore = Datastore.get(ws, "workspaceblobstore")
    
    logical_genome = [i == 1 for i in genome]
    tmp_columns = columns[logical_genome]
    if "Label" not in tmp_columns:
        tmp_columns = np.append(tmp_columns, ["Label"])
    
    Dataset.Tabular.register_pandas_dataframe(pd.DataFrame(data=[tmp_columns]), datastore, "dataset_test", show_progress=True)
    

def print_stats(result: pd.DataFrame):
    print("\n\n == Statystyki == \n")
    print("Accuracy: {0:.4f}".format(result.loc[0, 'accuracy']))
    print("Precision score: {0:.4f}".format(result.loc[0, 'precision']))
    print("Recall score: {0:.4f}".format(result.loc[0, 'recall']))
    print("f1 score: {0:.4f}".format(result.loc[0, 'f1']))
    print("auc score: {0:.4f}".format(result.loc[0, 'auc']))
    print("\n ===================================== \n")