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


def get_occurrences(distances, edges):
    hist_val = np.array([])
    for x in distances:
        hist, _ = np.histogram(x, edges)
        #     print(i,hist)
        hist_val = np.vstack([hist_val, hist]) if hist_val.size else hist

    # TODO : put an "if" here if you want choose among
    #       balanced values or not
    for i in range(edges[1:].size):
        area = np.pi * (np.square(edges[1:][i]) - np.square(edges[1:][i - 1])) if i else np.pi * np.square(edges[1:][i])
        hist_val[:, i] = np.true_divide(hist_val[:, i], area)

    return hist_val


def time_plot_histogram(file_title, values, y_edges, alpha, rho, num_robots, storagePath):
    y_edges = y_edges.round(decimals=3)
    fig = plt.figure(figsize=(10, 5), dpi=160)
    plt.ylabel('Distance from origin')
    plt.xlabel('time(s)')
    plt.legend()
    yticks = y_edges
    # plt.imshow(distances,interpolation='none')
    ax = sns.heatmap(values, yticklabels=yticks, vmin=0, cmap="viridis")
    ax.set_title(
        "Robots diffusion with " + r"$\bf{Robots}$:" + num_robots + r" $\bf{\rho}:$" + rho + " and " + r"$\bf{\alpha}:$" + alpha)
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
    return (w_displacement_array)


#     wmsd = np.mean(w_displacement_array)


def plot_heatmap(dictionary, w_size, storage_dir):
    for key, value in dictionary.items():
        fig = plt.figure(figsize=(12, 8))
        dataFrame = pd.DataFrame.from_dict(value)
        reversed_df = dataFrame.iloc[::-1]
        ax = sns.heatmap(reversed_df, annot=True, fmt=".2e", vmin=0.0001, vmax=0.01, cmap="viridis")
        #qui magari metti un if, se il titolo esiste allora non va messo quello qua sotto
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


def plot_both_wmsd(windowed, base_matrix, total_wmsd_matrix, alpha, rho, num_robots, storage_dir):
    fig = plt.figure(figsize=(20, 10), dpi=160, facecolor='w', edgecolor='k')
    for i, y in enumerate(total_wmsd_matrix):
        if (windowed):
            times = np.arange(len(y)) * 10
        else:
            times = np.linspace(0, len(y) * (i + 1) * 10, len(y), endpoint=True)

        #         print("indice finestra: ", i+1)
        #         print("numero di punti: ", len(y))
        #         print("Max val linspace: ", len(y)*(i+1)*10)
        #         print("times shape:", times.shape)
        #         print(times)
        plt.plot(times, y, label=i + 1, marker='o', color=mapcolors[i])

    for i, y in enumerate(base_matrix):
        if (windowed):
            times = np.arange(len(y)) * 10
        else:
            times = np.linspace(0, len(y) * (i + 1) * 10, len(y), endpoint=True)

        plt.plot(times, y, label="b" + str(i + 1), linestyle='dashed', alpha=0.6, color=mapcolors[i])

    fig.legend(loc=7, bbox_to_anchor=(0.95, 0.5))  # , prop={'size': 20})
    #     fig.subplots_adjust(right=0.9)
    #     plt.show()

    plt.title(
        "WMSD with different w_size, with " + r"$\bf{Robots}:$" + num_robots + r" $\bf{\rho}:$" + rho + " and " + r"$\bf{\alpha}:$" + alpha)
    plt.ylabel('WMSD')
    plt.xlabel('time(s)')
    #     plt.legend(loc='lower right')
    plt.xticks(np.arange(0, 1900, 200))
    #     plt.grid(which='minor')
    plt.grid()
    plt.ylim((0, 0.01))

    # ax = plt.axes()
    # #     plt.setp(ax.get_xticklabels(),visible=False)
    #     # Make a plot with major ticks that are multiples of 20 and minor ticks that
    #     # are multiples of 5.  Label major ticks with '%d' formatting but don't label
    #     # minor ticks.
    #     ax.xaxis.set_major_locator(MultipleLocator(100))
    #     ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    #     # For the minor ticks, use no labels; default NullFormatter.
    #     ax.xaxis.set_minor_locator(MultipleLocator(10))
    #     ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    #     ax.yaxis.set_major_formatter(FormatStrFormatter('%f'))

    #     plt.show();
    fileName = "comparison_robots_%s_rho_%s_alpha_%s.png" % (num_robots, rho, alpha)
    plt.savefig(storage_dir + '/' + fileName)
    plt.close(fig)


#     plt.setp(ax2.get_xticklabels(), visible=False)
#     frame1 = plt.gca()
#     frame1.axes.label.#().set_visible(False)


def load_pd_positions(dirPath, experiment_type):
    if experiment_type != "experiment" and experiment_type != "baseline":
        print("experiment_type could be only $experiment or $baseline")
        exit(-1)

    num_experiment = len([name for name in os.listdir(dirPath) if
                          (os.path.isfile(os.path.join(dirPath, name)) and (name.endswith('position.tsv')))])

    if os.path.exists(dirPath + "/" + experiment_type + ".pkl"):
        return num_experiment, pd.read_pickle(dirPath + "/" + experiment_type + ".pkl")

    print("Generating pickle positions file in " + dirPath + "/" + experiment_type + ".pkl")
    df = pd.DataFrame()
    for filename in os.listdir(dirPath):
        if filename.endswith('position.tsv'):
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
    for k1,val1 in sorted(dictionary.items()):
        temp[k1] = dict()
        for k2, val2 in sorted(val1.items()):
            temp[k1][k2] = dict()
            for k3,val3 in sorted(val2.items()):
                temp[k1][k2][k3]=val3
    return temp

def generate_pdf(folder):
    if not os.path.isdir(folder):
        print_help()
        exit(-1)

    imagelist = []
    for dirName, subdirList, fileList in os.walk(folder):
        fileList.sort()
        for i in fileList:
            if (i.endswith(".png")):
                img = Image.open(dirName + '/' + i)
                img = img.convert('RGB')
                imagelist.append(img)

    imagelist[0].save(folder + '/plots.pdf', save_all=True, append_images=imagelist)
