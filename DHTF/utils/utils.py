import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import os
from scipy import stats
import math
from scipy.optimize import curve_fit
from scipy.stats import norm
import seaborn as sns
import pandas as pd
from PIL import Image
from natsort import natsorted


def print_help():
    print("usage : folder_path")


# colors = ['red','blue','darkgreen','crimson','turquoise', 'khaki','navy', 'orangered', 'sienna']
Ncolors = 9
colormap = plt.cm.viridis  # LinearSegmentedColormap
Ncolors = min(colormap.N, Ncolors)
mapcolors = [colormap(int(x * colormap.N / Ncolors)) for x in range(Ncolors)]


def load_pd_positions(dirPath, experiment_type):
    if experiment_type != "experiment" and experiment_type != "baseline":
        print("experiment_type could be only $experiment or $baseline")
        exit(-1)

    num_experiment = len([name for name in os.listdir(dirPath) if
                          (os.path.isfile(os.path.join(dirPath, name)) and (name.endswith('position.tsv')))])

    if os.path.exists(dirPath + "/" + experiment_type + ".pkl"):
        # print("Loading pickle positions file in " + dirPath + "/" + experiment_type + ".pkl")
        return num_experiment, pd.read_pickle(dirPath + "/" + experiment_type + ".pkl")
    # else:
    #     print("Baseline:"+dirPath+" not an existing path")
    #     exit(-1)

    print("Generating pickle positions file in " + dirPath + "/" + experiment_type + ".pkl")
    df = pd.DataFrame()
    for filename in os.listdir(dirPath):
        if filename.endswith('position.tsv'):
            if not os.path.getsize(os.path.join(dirPath, filename)) > 0:
                print("Error, empty file at:" + os.path.join(dirPath, filename))
                continue
            df_single = pd.read_csv(dirPath + "/" + filename, sep="\t")
            df = df.append(df_single)

    df.to_pickle(dirPath + "/" + experiment_type + ".pkl")
    return num_experiment, df


def load_pd_completed_tasks(dirPath, mode="client"):
    num_experiment = len([name for name in os.listdir(dirPath) if
                          (os.path.isfile(os.path.join(dirPath, name)) and (name.endswith('position.tsv')))])

    if os.path.exists(dirPath + "/completed_task_" + mode + ".pkl"):
        print("Loading pickle positions file in " + dirPath + "/completed_task_" + mode + ".pkl")
        return num_experiment, pd.read_pickle(dirPath + "/completed_task_" + mode + ".pkl")

    print("Generating pickle positions file in " + dirPath + "/completed_task_" + mode + ".pkl")
    df = pd.DataFrame()
    for filename in os.listdir(dirPath):
        if filename.endswith('taskLOG'+mode+'.tsv'):
            if not os.path.getsize(os.path.join(dirPath, filename)) > 0:
                print("Error, empty file at:" + os.path.join(dirPath, filename))
                continue
            df_single = pd.read_csv(dirPath + "/" + filename, sep="\t")
            df = df.append(df_single)

    df.to_pickle(dirPath + "/" + experiment_type + ".pkl")
    return num_experiment, df


def load_pd_times(dirPath, experiment_type):
    if experiment_type != "times":
        print("experiment_type could be only $times")
        exit(-1)

    num_experiment = len([name for name in os.listdir(dirPath) if
                          (os.path.isfile(os.path.join(dirPath, name)) and (name.endswith('position.tsv')))])

    if os.path.exists(dirPath + "/" + experiment_type + ".pkl"):
        return num_experiment, pd.read_pickle(dirPath + "/" + experiment_type + ".pkl")

    print("Generating pickle times file")
    df = pd.DataFrame()
    for filename in os.listdir(dirPath):
        if filename.endswith('time_results.tsv'):
            if not os.path.getsize(os.path.join(dirPath, filename)) > 0:
                print("Error, empty file at:" + os.path.join(dirPath, filename))
                continue
            df_single = pd.read_csv(dirPath + "/" + filename, sep="\t")
            df = df.append(df_single)

    df.to_pickle(dirPath + "/" + experiment_type + ".pkl")
    return num_experiment, df


def sort_nested_dict(dictionary):
    temp = dict()
    for k1, val1 in sorted(dictionary.items()):
        temp[k1] = dict()
        for k2, val2 in sorted(val1.items()):
            temp[k1][k2] = dict()
            for k3, val3 in sorted(val2.items()):
                temp[k1][k2][k3] = val3
    return temp


def generate_pdf(folder):
    if not os.path.isdir(folder):
        print_help()
        exit(-1)

    imagelist = []
    for dirName, subdirList, fileList in os.walk(folder):
        for i in natsorted(fileList):
            if i.endswith(".png"):
                img = Image.open(dirName + '/' + i)
                img = img.convert('RGB')
                imagelist.append(img)

    imagelist[0].save(folder + os.path.basename(os.path.normpath(folder)) + '.pdf', save_all=True,
                      append_images=imagelist)
