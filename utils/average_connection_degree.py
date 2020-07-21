from utils import utils
from utils import config

import os
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Ncolors = config.population_size
colormap = plt.cm.viridis  # LinearSegmentedColormap
Ncolors = min(colormap.N, Ncolors)
mapcolors = [colormap(int(x * colormap.N / Ncolors)) for x in range(Ncolors)]


def get_connections(positions):
    connection_number_history = np.zeros((positions.shape[0], positions.shape[1]))
    for index, pos_t in enumerate(positions.transpose(1, 0, 2)):
        connection_number = np.zeros(positions.shape[0])
        #     print(c.shape)
        for idx, elem in enumerate(pos_t):
            #         print('\t',elem)
            robot_distance = np.sqrt(np.sum((pos_t - elem) ** 2, axis=1))
            #         print(robot_distance, end='\n\n\n')
            connection_number[idx] = robot_distance[np.where(robot_distance <= 0.1)].size - 1
        connection_number_history[:, index] = connection_number
    return connection_number_history

def connection_heatmap(edges, time_connections, alpha, rho, num_robots, avg_connection_degree_dir):
    y_edges = edges[1:].round(decimals=3)
    fig = plt.figure(figsize=(10, 5), dpi=160)
#     plt.ylabel('distance from the origin')
#     plt.xlabel('time(s)')
#     plt.legend()
    yticks = y_edges
    ax = sns.heatmap(time_connections, yticklabels=yticks, cmap="viridis")
    ax.set_title(
        "Average connection from the origin with " + r"$\bf{Robots}$:" + num_robots + r" $\bf{\rho}:$" + rho + " and " + r"$\bf{\alpha}:$" + alpha)
    file_name = "average_connection_heatmap_robots_%s_rho_%s_alpha_%s.png" % (num_robots, rho, alpha)
    plt.savefig(avg_connection_degree_dir + '/' + file_name)
    plt.close(fig)
#     ax.set_title('Average connection from the origin')
#     plt.show()

# Average connection degree distribuited over different circular section positions (using Heatmaps)
def avg_connection_degree_heatmap(folder_experiment, avg_connection_degree_dir):


    for dirName, subdirList, fileList in os.walk(folder_experiment):

        num_robots = "-1"
        rho = -1.0
        alpha = -1.0
        elements = dirName.split("_")
        for e in elements:
            if e.startswith("robots"):
                num_robots = e.split("#")[-1]
            if (e.startswith("rho")):
                rho = float(e.split("#")[-1])
            if (e.startswith("alpha")):
                alpha = float(e.split("#")[-1])

        #         print(num_robots+' '+str(rho)+' '+str(alpha))
        if (num_robots == "-1" or rho == -1.0 or alpha == -1):
            continue
        else:
            # print(dirName)
            runs = len([f for f in fileList if f.endswith('position.tsv')])
        #         print(runs)

        rho_str = str(rho)
        alpha_str = str(alpha)

        # POSIZIONI
        [_, df_experiment] = utils.load_pd_positions(dirName, "experiment")

        positions_concatenated = df_experiment.values[:, 1:]
        [num_robot, num_times] = positions_concatenated.shape
        positions_concatenated = np.array([x.split(',') for x in positions_concatenated.ravel()], dtype=float)
        positions_concatenated = positions_concatenated.reshape(num_robot, num_times, 2)

        position_concatenated_split = np.split(positions_concatenated, runs)

        # print("positions_concatenated.shape: ", positions_concatenated.shape)

        # CONNECTIONS
        connection_number_history = np.array([])
        for single_run in position_concatenated_split:
            connection_number_history = np.vstack((connection_number_history, get_connections(
                single_run))) if connection_number_history.size else get_connections(single_run)

        # print(connection_number_history.shape)

        # ORIGIN DISTANCE
        origin_distance = utils.distance_from_the_origin(positions_concatenated).T

        connection_in_time = np.ones((config.bin_edges.size - 1, origin_distance.shape[1])) * -1

        for idx, distance_t in enumerate(origin_distance.T):
            #     print(distance_t)
            for edge_idx in range(config.bin_edges.size - 1):
                #         print(bin_edges[edge_idx],bin_edges[edge_idx+1])
                #         print("\t",edge_idx)
                
                where_index = np.where(
                    np.logical_and(distance_t >= config.bin_edges[edge_idx], distance_t < config.bin_edges[edge_idx + 1]))
                connection_in_time[edge_idx, idx] = np.mean(connection_number_history[where_index])

        connection_heatmap(config.bin_edges, connection_in_time, alpha_str, rho_str, num_robots, avg_connection_degree_dir)

# Average connection plots among different population sizes
def avg_connection_plot_different_population_sizes(folder_experiment, avg_connection_degree_dir):

    for a in config.alpha_array:
        for r in config.rho_array:
            #         print("a",a,"r",r)
            robot_arr = []
            fig = plt.figure(figsize=(20, 10), dpi=80)

            #         for dirName, subdirList, fileList in os.walk(folder_experiment):
            for dirName in natsorted(os.listdir(folder_experiment)):
                dirPath = os.path.join(folder_experiment, dirName)

                num_robots = "-1"
                elements = dirName.split("_")
                for e in elements:
                    if e.startswith("robots"):
                        num_robots = e.split("#")[-1]
                    if (e.startswith("rho")):
                        rho = float(e.split("#")[-1])
                    if (e.startswith("alpha")):
                        alpha = float(e.split("#")[-1])

                #         print(num_robots+' '+str(rho)+' '+str(alpha))
                #             print(alpha, rho)
                if (num_robots == "-1" or rho != r or alpha != a):
                    continue
                else:
                    #                 print(dirName)

                    robot_arr += [int(num_robots)]
                    #                 print(num_robots, int(num_robots))
                    runs = len([f for f in os.listdir(dirPath) if
                                (os.path.isfile(os.path.join(dirPath, f)) and f.endswith('position.tsv'))])
                #         print(runs)

                rho_str = str(rho)
                alpha_str = str(alpha)

                [_, df_experiment] = utils.load_pd_positions(dirPath, "experiment")

                positions_concatenated = df_experiment.values[:, 1:]
                [num_robot, num_times] = positions_concatenated.shape
                positions_concatenated = np.array([x.split(',') for x in positions_concatenated.ravel()], dtype=float)
                positions_concatenated = positions_concatenated.reshape(num_robot, num_times, 2)

                position_concatenated_split = np.split(positions_concatenated, runs)

                connection_number_history = np.array([])
                for single_run in position_concatenated_split:
                    connection_number_history = np.vstack((connection_number_history, get_connections(
                        single_run))) if connection_number_history.size else get_connections(single_run)

                connection_number_history_mean = np.mean(connection_number_history, axis=0)
                #             print(num_times)
                #             print(connection_number_history_mean.shape)
                #             print(len(robot_arr))
                plt.plot(np.arange(num_times), connection_number_history_mean, linewidth=2, label=num_robots,
                         color=mapcolors[len(robot_arr)])

            plt.title("Average connection with " + r" $\bf{\rho}:$" + rho_str + " and " + r"$\bf{\alpha}:$" + alpha_str)
            plt.ylabel('mean connection link')
            plt.xlabel('time')
            #     plt.legend(loc='lower right')
            if not config.open_space_flag:
                plt.yticks(np.arange(0, 9, 0.5))
            else:
                plt.yticks(np.arange(0, 20, 0.5))
            #     plt.grid(which='minor')
            plt.grid()
            plt.legend(loc=2)  # , bbox_to_anchor=(0.95, 0.5))
            file_name = "average_connection_plot_rho_%s_alpha_%s.png" % (rho_str, alpha_str)
            plt.savefig(avg_connection_degree_dir + '/' + file_name)
            plt.close(fig)
    #         plt.show()
