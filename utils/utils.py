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
    print("usage : folder_path, window size (1 for 10, 2 for 20, ....)")


def distance_from_the_origin(df_values):
    distances = np.array([])
    # df_values1 = df_values[:3,:4]
    for i in range(df_values.shape[1]):
        #     print(i)
        x_pos = df_values[:, i, 0]
        y_pos = df_values[:, i, 1]
        distances = np.vstack([distances, np.sqrt(x_pos ** 2 + y_pos ** 2)]) if distances.size else np.sqrt(
            x_pos ** 2 + y_pos ** 2)
    # print(distances)
    # print("distances shape", distances.shape)
    return distances


def get_occurrences(distances, edges, runs):
    hist_val = np.array([])
    for x in distances:
        hist, _ = np.histogram(x, edges)
        #     print(i,hist)
        hist_val = np.vstack([hist_val, hist]) if hist_val.size else hist

    for i in range(edges[1:].size):
        area = np.pi * (np.square(edges[1:][i]) - np.square(edges[1:][i - 1])) if i else np.pi * np.square(edges[1:][i])
        hist_val[:, i] = np.true_divide(hist_val[:, i], area*runs)

    return hist_val


def time_plot_histogram(values, y_edges, alpha, rho, num_robots, storagePath):
    y_edges = y_edges.round(decimals=3)
    fig = plt.figure(figsize=(10, 5), dpi=160)
    # plt.ylabel('Distance from origin')
    # plt.xlabel('time(s)')
    # plt.legend()
    yticks = y_edges
    # plt.imshow(distances,interpolation='none')

    # print("num_robots:", num_robots, end="")
    if num_robots == "10":
        v_max = 25
    elif num_robots == "20":
        v_max = 50
    elif num_robots == "50":
        v_max = 120
    elif num_robots == "100":
        v_max = 200
    else:
        print("Type", type(num_robots))

    # print("\t v_max:", v_max)
    ax = sns.heatmap(values, yticklabels=yticks, cmap="viridis", vmin=0, vmax=v_max)
    ax.set_title(
        "Robots diffusion with " + r"$\bf{Robots}$:" + num_robots + r" $\bf{\rho}:$" + rho + " and " + r"$\bf{\alpha}:$" + alpha)
    ax.set_ylabel('distance from the origin')
    ax.set_xlabel('time')
    file_name = "dist_heat_robots_%s_rho_%s_alpha_%s.png" % (num_robots, rho, alpha)
    plt.savefig(storagePath + '/' + file_name)
    plt.close(fig)


'''Windowed mean square displacement'''
'''Input : dataFrame (num_robot x stored times), window size'''
'''Output : average wmsd for all the robot at each timestep'''


def window_displacement(df, window_size):
    # print(df.shape[1])
    w_displacement_matrix = np.array([])
    for f in range(window_size, df.shape[1]):
        xf = df[:, f]
        xi = df[:, f - window_size]
        sq_distance = np.sum((xf - xi) ** 2, axis=1)
        wmsd = np.true_divide(sq_distance, window_size ** 2)  # wmsd for the single robots
        w_displacement_matrix = np.column_stack([w_displacement_matrix, wmsd]) if w_displacement_matrix.size else wmsd
    #         print(f-window_size, f)
    w_displacement_array = np.mean(w_displacement_matrix, axis=0)
    return (w_displacement_array)


'''Fixed window mean square displacement'''
'''Input : dataFrame (num_robot x stored times), window size'''
'''Output : average fixed wmsd for all the robot at each timestep'''


def fixed_window_displacement(df, window_size):
    w_displacement_matrix = np.array([])
    for f in range(window_size, df.shape[1], window_size):
        tf = df[:, f]
        ti = df[:, f - window_size]
        sq_distance = np.sum((tf - ti) ** 2, axis=1)
        wmsd = np.true_divide(sq_distance, window_size ** 2)
        w_displacement_matrix = np.column_stack([w_displacement_matrix, wmsd]) if w_displacement_matrix.size else wmsd

    w_displacement_array = np.mean(w_displacement_matrix, axis=0)
    return w_displacement_array


def time_mean_square_displacement(df):
    tsd_matrix = np.array([])   # time square displacement (mean performed at the end)
    x0 = df[:, 0]
    for t in range(1, df.shape[1]):
        xt = df[:, t]
        sq_disp = np.sum((xt - x0) ** 2, axis=1)
        tsd_matrix = np.column_stack([tsd_matrix, sq_disp]) if tsd_matrix.size else sq_disp

    return np.mean(tsd_matrix, axis=0)


def plot_heatmap(dictionary, w_size, storage_dir):
    for key, value in dictionary.items():
        fig = plt.figure(figsize=(12, 8))
        dataFrame = pd.DataFrame.from_dict(value)
        reversed_df = dataFrame.iloc[::-1]
        ax = sns.heatmap(reversed_df, annot=True, fmt=".2e", vmin=0.0001, vmax=0.01, cmap="viridis")
        # qui magari metti un if, se il titolo esiste allora non va messo quello qua sotto
        ax.set_title("Heatmap of WMSD for %s robots, w_size:%s" % (key, w_size))
        ax.set_ylabel("alpha")
        ax.set_xlabel("rho")
        #         plt.show();
        # Salva su file
        file_name = "WMSD_%s_robots_wsize_%s_heatmap.png" % (key, w_size)
        plt.savefig(storage_dir + '/' + file_name)
        plt.close(fig)


# colors = ['red','blue','darkgreen','crimson','turquoise', 'khaki','navy', 'orangered', 'sienna']
Ncolors = 9
colormap = plt.cm.viridis  # LinearSegmentedColormap
Ncolors = min(colormap.N, Ncolors)
mapcolors = [colormap(int(x * colormap.N / Ncolors)) for x in range(Ncolors)]


def plot_both_wmsd(base_matrix, total_wmsd_matrix, alpha, rho, num_robots, storage_dir, windowed=True, title="TMSD"):
    fig = plt.figure(figsize=(20, 10), dpi=160, facecolor='w', edgecolor='k')
    for i, y in enumerate(total_wmsd_matrix):
        if (windowed):
            times = np.arange(len(y)) * 10
        else:
            times = np.linspace(0, len(y) * (i + 1) * 10, len(y), endpoint=True)

        plt.plot(times, y, label=i + 1, marker='o', color=mapcolors[i])

    for i, y in enumerate(base_matrix):
        if (windowed):
            times = np.arange(len(y)) * 10
        else:
            times = np.linspace(0, len(y) * (i + 1) * 10, len(y), endpoint=True)

        plt.plot(times, y, label="b" + str(i + 1), linestyle='dashed', alpha=0.6, color=mapcolors[i])

    fig.legend(loc=7, bbox_to_anchor=(0.95, 0.5))
    #     plt.show()

    plt.title(title + ", with " + r"$\bf{Robots}:$" + num_robots + r" $\bf{\rho}:$" + rho + " and " + r"$\bf{\alpha}:$" + alpha)
    plt.ylabel(title)
    plt.xlabel('time(s)')
    # plt.xticks(np.arange(0, 1900, 200))
    plt.grid()
    plt.ylim(bottom=0.0, top=40)

    #     plt.show();
    fileName = "comparison_robots_%s_rho_%s_alpha_%s.png" % (num_robots, rho, alpha)
    plt.savefig(storage_dir + '/' + fileName)
    plt.close(fig)


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


def load_pd_times(dirPath, experiment_type):
    if (experiment_type != "times"):
        print("experiment_type could be only $times")
        exit(-1)

    num_experiment = len([name for name in os.listdir(dirPath) if
                          (os.path.isfile(os.path.join(dirPath, name)) and (name.endswith('position.tsv')))])

    if (os.path.exists(dirPath + "/" + experiment_type + ".pkl")):
        return (num_experiment, pd.read_pickle(dirPath + "/" + experiment_type + ".pkl"))

    print("Generating pickle times file")
    df = pd.DataFrame()
    for filename in os.listdir(dirPath):
        if filename.endswith('time_results.tsv'):
            df_single = pd.read_csv(dirPath + "/" + filename, sep="\t")
            df = df.append(df_single)

    df.to_pickle(dirPath + "/" + experiment_type + ".pkl")
    return (num_experiment, df)


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
            if (i.endswith(".png")):
                img = Image.open(dirName + '/' + i)
                img = img.convert('RGB')
                imagelist.append(img)

    imagelist[0].save(folder + os.path.basename(os.path.normpath(folder)) + '.pdf', save_all=True,
                      append_images=imagelist)
