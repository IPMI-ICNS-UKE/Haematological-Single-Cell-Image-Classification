import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_ssl():
    path = "/home/crohling/Documents/cell_classification/unsupervised_dino_bm_25"
    ba = np.zeros((4,4))
    ba_std = np.zeros((4,4))
    names = ["blood_acevedo", "blood_matek", "blood_raabin", "bm"]
    for name in names:
        for name_2 in names:
            BA = []
            for i in range(25):
                file = "fit_" + name +  "_eval_" + name_2
                path_cm = path + "/" + file + "/"+ os.listdir(path + "/" + file)[0] + "/evaluation/cmfs/"+ f"confusion_matrix_SVC_100_iter_{i}.csv"

                cm = (pd.read_csv(path_cm).to_numpy()[:,1:]).astype(float)
                BA.append(cal_bal_acc(cm))

            ba[nameToId(name), nameToId(name_2)] = np.mean(BA)
            ba_std[nameToId(name), nameToId(name_2)] = np.std(BA)
    visualize(ba, ba_std, "DINO")


def visualize_supervised():
    path = "/home/crohling/Documents/cell_classification/supervised_second"
    ba = np.zeros((4,4))
    ba_std = np.zeros((4,4))
    names = ["blood_acevedo", "blood_matek", "blood_raabin", "bm"]
    for name in names:
        for name_2 in names:
            file = "train_" + name +  "_eval_" + name_2
            cfms = []
            BA = []
            for i in range(3):
                path_results = path + "/" + file + "/"+ os.listdir(path + "/" + file)[i] + "/confusion_matrix.csv"
                cmpd = pd.read_csv(path_results)

                conv = cmpd.to_numpy()[:,1:]
                BA.append(cal_bal_acc(conv))
                cfms.append(np.array(conv))

            cfms = np.array(cfms)
            cm = np.mean(cfms.astype(float), axis=0)
            cm_std = np.std(cfms.astype(float), axis=0)
            class_names = cmpd.to_numpy()[:,0]
            # os.makedirs("/home/crohling/Documents/cell_classification/Final_Data/Supervised" + f"/fit_{name}_eval_{name_2}")
            cm_mean_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            cm_mean_df.to_csv("/home/crohling/Documents/cell_classification/Final_Data/Supervised" + f'/fit_{name}_eval_{name_2}/confusion_matrix_fit_{name}_eval_{name_2}_mean.csv', index=True, header=True)
            cm_mean_df = pd.DataFrame(cm_std, index=class_names, columns=class_names)
            cm_mean_df.to_csv(
                "/home/crohling/Documents/cell_classification/Final_Data/Supervised" + f'/fit_{name}_eval_{name_2}/confusion_matrix_fit_{name}_eval_{name_2}_std.csv',
                index=True, header=True)


            ba[nameToId(name), nameToId(name_2)] = np.mean(BA)
            ba_std[nameToId(name), nameToId(name_2)] = np.std(BA)

    visualize(ba, ba_std, "Supervised")



def cal_bal_acc(cm):

    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    BA = np.mean(TPR)
    return BA



def check_shape():
    path = "/home/crohling/Documents/cell_classification/unsupervised_dino_bm_25"
    path_sup = "/home/crohling/Documents/cell_classification/supervised_second"
    names = ["blood_acevedo", "blood_matek", "blood_raabin", "bm"]
    for name in names:
        for name_2 in names:
            file = "fit_" + name +  "_eval_" + name_2
            file_sup = "train_" + name + "_eval_" + name_2
            path_results_sup = path_sup + "/" + file_sup + "/" + os.listdir(path_sup + "/" + file_sup)[0] + "/confusion_matrix.csv"
            path_results = path + "/" + file + "/"+ os.listdir(path + "/" + file)[0] + "/evaluation/cmfs/confusion_matrix_SVC_100_iter_24.csv"
            csv = pd.read_csv(path_results).to_numpy()
            csv_2 = pd.read_csv(path_results_sup).to_numpy()
            print(name, name_2,np.sort(csv[:,0]) == np.sort(csv_2[:,0]), np.sort(csv[:,0]), np.sort(csv_2[:,0]))

def visualize(ba, ba_std, name):
    names = ["blood_acevedo", "blood_matek", "blood_raabin", "bm"]
    fig, ax = plt.subplots()
    im = ax.imshow(ba, cmap="RdYlGn", vmin=0, vmax=1)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(names)), labels=names)
    ax.set_yticks(np.arange(len(names)), labels=names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(names)):
        for j in range(len(names)):
            text = ax.text(j, i, "{:.2f} ".format(ba[i, j]) + u"\u00B1" + " {:.2f}".format(ba_std[i, j]),
                           ha="center", va="center", color="black", size=8)

    ax.set_title("Mean balanced accuracy " + name)
    fig.tight_layout()
    plt.show()

    ba_mean = pd.DataFrame(ba, index=names, columns=names)
    ba_mean.to_csv(
        "/home/crohling/Documents/cell_classification/Final_Data/"+ name + f"/balanced_accuracy_{name}_mean.csv")
    ba_std = pd.DataFrame(ba_std, index=names, columns=names)
    ba_std.to_csv(
        "/home/crohling/Documents/cell_classification/Final_Data/"+ name + f"/balanced_accuracy_{name}_std.csv")


def nameToId(name):
    names = ["blood_acevedo", "blood_matek", "blood_raabin", "bm"]
    dict = {class_name: idx for idx, class_name in enumerate(names)}
    return dict[name]


if __name__ == "__main__":
    visualize_ssl()
    visualize_supervised()
