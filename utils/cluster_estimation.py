from utils import utils
from utils import config

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import math


def check_connection(robot_pos_1, robot_pos_2):
    dist = np.linalg.norm(robot_pos_1 - robot_pos_2)
    if dist <= config.distance_parameter * config.kilo_diameter:
        return 1
    else:
        return 0


def get_connected_components(single_run):
    n_clusters = np.zeros(single_run.shape[1], dtype=int)
    biggest_clusters_per_time = np.zeros(single_run.shape[1], dtype=int)
    for c in range(single_run.shape[1]):
        xi = single_run[:, c, :]
        adjacency_matrix = np.zeros([xi.shape[0], xi.shape[0]], dtype=int)
        for i, ki in enumerate(xi):
            for j, kj in enumerate(xi[i + 1:]):
                #         if(check_connection(ki,kj)):
                #             print(i,i+j+1,'\t',check_connection(ki,kj))
                adjacency_matrix[i, i + j + 1] = check_connection(ki, kj)
                adjacency_matrix[i + j + 1, i] = check_connection(ki, kj)
        #     print(adjacency_matrix)
        csr_adjacency_matrix = csr_matrix(adjacency_matrix)  # cambiare nome
        n_cluster, cluster_labels = connected_components(csgraph=csr_adjacency_matrix, directed=False,
                                                         return_labels=True)
        n_clusters[c] = n_cluster
        biggest_clusters_per_time[c] = np.max(np.bincount(cluster_labels))
    #     print("timestep %d : num_components %d" %(c, n_cluster))

    return biggest_clusters_per_time, n_clusters


def plot_info_clusters(biggest_clusters_avg, n_clusters_avg, rho, alpha, num_robots, store_dir):
    fig = plt.figure(figsize=(20, 10), dpi=160, facecolor='w', edgecolor='k')

    times = np.arange(n_clusters_avg.size) * 10

    # colors = ['red','blue','darkgreen','crimson','turquoise', 'khaki','navy', 'orangered', 'sienna']
    Ncolors = 2
    colormap = plt.cm.viridis  # LinearSegmentedColormap
    Ncolors = min(colormap.N, Ncolors)
    mapcolors = [colormap(int(x * colormap.N / Ncolors)) for x in range(Ncolors)]

    # biggest_clusters_avg
    plt.subplot(211)
    plt.plot(times, biggest_clusters_avg, marker='.', color=mapcolors[0])
    yint = range(math.floor(min(biggest_clusters_avg)) - 1, math.ceil(max(biggest_clusters_avg)) + 1)
    plt.yticks(yint)
    plt.ylabel('biggest cluster avg')
    plt.xlabel('time (s)')

    # n_clusters_avg
    plt.subplot(212)
    plt.plot(times, n_clusters_avg, marker='.', color=mapcolors[0])
    yint = range(math.floor(min(n_clusters_avg)) - 1, math.ceil(max(n_clusters_avg)) + 1)
    plt.yticks(yint)
    plt.ylabel('Number of cluster avg')
    plt.xlabel('time (s)')

    plt.suptitle('Cluster evolution with ' + r'$\bf{Robots}$:' + num_robots + r' $\bf{\rho}:$' + rho + ' and '
                 + r'$\bf{\alpha}:$' + alpha, fontsize=25)
    file_name = 'cluster_evolution_robots_%s_rho_%s_alpha_%s.png' % (num_robots, rho, alpha)
    # plt.show()
    plt.savefig(store_dir + '/' + file_name)
    plt.close(fig)


def cluster_estimation_study(folder_experiment, cluster_dir):
    for dirName, subdirList, fileList in os.walk(folder_experiment):
        biggest_clusters_avg = np.array([])
        n_clusters_avg = np.array([])

        num_robots = "-1"
        rho = -1.0
        alpha = -1.0
        elements = dirName.split("_")
        for e in elements:
            if e.startswith("robots"):
                num_robots = e.split("#")[-1]
            if e.startswith("rho"):
                rho = float(e.split("#")[-1])
            if e.startswith("alpha"):
                alpha = float(e.split("#")[-1])

        #         print(num_robots+' '+str(rho)+' '+str(alpha))
        if num_robots == "-1" or rho == -1.0 or alpha == -1:
            continue
        else:
            print(dirName)
            runs = len([f for f in fileList if f.endswith('position.tsv')])
        #         print(runs)

        [_, df_experiment] = utils.load_pd_positions(dirName, "experiment")
        positions_concatenated = df_experiment.values[:, 1:]  # [robots, times]
        [num_robot, num_times] = positions_concatenated.shape
        positions_concatenated = np.array([x.split(',') for x in positions_concatenated.ravel()], dtype=float)
        positions_concatenated = positions_concatenated.reshape(num_robot, num_times, 2)
        position_concatenated_split = np.split(positions_concatenated, runs)

        for single_run in position_concatenated_split:
            #         print('single run processing')
            biggest_clusters_per_time, n_clusters = get_connected_components(single_run)
            biggest_clusters_avg = np.vstack([biggest_clusters_avg,
                                              biggest_clusters_per_time]) if biggest_clusters_avg.size else biggest_clusters_per_time
            n_clusters_avg = np.vstack([n_clusters_avg, n_clusters]) if n_clusters_avg.size else n_clusters

        biggest_clusters_avg = np.mean(biggest_clusters_avg, axis=0)
        n_clusters_avg = np.mean(n_clusters_avg, axis=0)

        print('Plotting')
        plot_info_clusters(biggest_clusters_avg, n_clusters_avg, str(rho), str(alpha), num_robots, cluster_dir)
