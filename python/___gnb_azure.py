import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from helper import read_b
import matplotlib.pyplot as plt
import mlflow


def simple_gnb(dataset: pd.DataFrame, dataset_test: pd.DataFrame):
    """
    Metoda wykorzystująca moduł sklearn, aby za pomocą Gaussian Naive Bayes wytrenować model za pomocą dataset.
    Następnie za pomocą dataset_test, przetestować model. Metoda zwraca wszystkie parametry związane z obliczeniami:
    [dokładność, precyzję, czułość, ocena, największa wykorzystana pamięć, czas trwania]

    :param dataset:
    :param dataset_test:
    :return:
    """

    # --- Wstępne informacje
    mlflow.sklearn.autolog()
    print("\n")
    print("=============================")
    print("=== Klasyfikowanie danych ===")
    print("=============================\n")

    print("\n == Wielkość plików dataset == \n")
    dataSize = read_b(dataset.memory_usage(index=False, deep=True).sum())
    dataTestSize = read_b(dataset_test.memory_usage(index=False, deep=True).sum())

    print("Dane dla modelu")
    print("\tRozmiar: {0}".format(dataSize))
    print("\tIlość wierszy: {0}".format(dataset.shape[0]))

    print("Dane testowe")
    print("\tRozmiar: {0}".format(dataTestSize))
    print("\tIlość wierszy: {0}".format(dataset_test.shape[0]))

    # --- Proces GNB --
    print("\n\n == Proces == \n")
    
    # --- Usunięcie NaN za pomocą 0
    print("Zamiana Nan oraz Inf na 0")

    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.fillna(0)

    dataset_test = dataset_test.replace([np.inf, -np.inf], np.nan)
    dataset_test = dataset_test.fillna(0)

    # --- Zamiana etykiet na 1 i 0
    print("Obróbka danych")

    # --- Przygotowanie X i Y gdzie X to dane, a Y to etykiety
    x = dataset.drop(["Label"], axis=1)
    y = dataset.loc[:, "Label"].values

    x_test = dataset_test.drop(["Label"], axis=1)
    y_test = dataset_test.loc[:, "Label"].values

    x.replace([np.inf, -np.inf], np.nan)
    x_test.replace([np.inf, -np.inf], np.nan)

    x = x.fillna(0)
    x_test = x_test.fillna(0)

    # ---Tworzenie GaussianNB oraz uczenie
    print("Trenowanie danych")

    nb = GaussianNB()
    nb.fit(x, y)

    # --- Obliczenie predykcji
    print("Obliczanie predykcji")
    y_pred = nb.predict(x_test)
    y_prob = nb.predict_proba(x_test)[:, 1]
    # --- Obliczanie metryk
    print("Obliczanie metryk")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = 2 * prec * rec / (rec + prec)
    auc = roc_auc_score(y_test, y_prob)

    # mlflow.autolog()
     
    print("Zakończenie procesu")
    
    # fpr, tpr, _ = roc_curve(y_test,  y_prob)
    print("wykresy")
    # create ROC curve
    # plt.plot(fpr,tpr)
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # fig = plt.savefig()
    # mlflow.log_figure(fig, "roc_curve.png")

    # confusion_matrix_plt = confusion_matrix(y_test, y_pred)
    # cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_plt, display_labels = [False, True])
    # cm_display.plot()
    # cm_disp = plt.savefig()
    # mlflow.log_figure(cm_disp, "matrix_curve.png")

    mlflow.log_metric('Accuracy', acc)
    mlflow.log_metric('Precision', prec)
    mlflow.log_metric('Recall', rec)
    mlflow.log_metric('F1', f1)
    mlflow.log_metric('AUC', auc)
    
    
    return pd.DataFrame({
        'accuracy': [acc],
        'precision': [prec],
        'recall': [rec],
        'f1': [f1],
        'auc': [auc]
    }, dtype=np.float)
