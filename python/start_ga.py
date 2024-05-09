from ga import ga_simple
from gnb import simple_gnb
import pandas as pd
import numpy as np
import glob 
import os

# dataset = pd.read_csv("C:\\Users\\Bartosz\\Documents\\praca-magisterska\\dane\\cicd\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
# dataset.columns = dataset.columns.str.strip()
# dataset_test = pd.read_csv("C:\\Users\\Bartosz\\Documents\\praca-magisterska\\dane\\cicd\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
# dataset_test.columns = dataset_test.columns.str.strip()
all_files = glob.glob(os.path.join('C:\\Users\\Bartosz\\Documents\\praca-magisterska\\dane\\cicd', "*.csv"))

dataset_test_1 = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
# dataset_test_1.columns = dataset_test_1.columns.str.strip()

dataset_test_1.to_csv("./dataset_test_all.csv", index=False)
# print(dataset_test_1.info(verbose=True))

# columns = pd.read_csv("./columns.txt")
# columns.columns = columns.columns.str.strip()

# dataset.loc[:, "Label"] = [1 if i == "BENIGN" else 0 for i in dataset.loc[:, "Label"]]
# dataset_test.loc[:, "Label"] = [1 if i == "BENIGN" else 0 for i in dataset_test.loc[:, "Label"]]
# dataset_test_1.loc[:, "Label"] = [1 if i == "BENIGN" else 0 for i in dataset_test_1.loc[:, "Label"]]

# col_new = ga_simple(dataset1=dataset, dataset2=dataset_test)
# col_new = np.strings.strip(col_new)
# print(columns.size)

# print(simple_gnb(dataset=dataset[columns.columns], dataset_test=dataset_test[columns.columns]))
# print(simple_gnb(dataset=dataset[columns.columns], dataset_test=dataset_test_1[columns.columns]))

