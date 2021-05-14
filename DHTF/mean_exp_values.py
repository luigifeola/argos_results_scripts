import os
from utils import boxplots_utils
import pandas as pd
from termcolor import colored
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from natsort import natsorted
import sys
from utils import boxplots_utils

# Display pandas df without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Numpy array printed with higher width
# np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

# experiment = "four_regions"
experiment = "random_mixed"

# walk = 'adaptive'
# walk = 'persistent'
walk = 'brownian'

groupSize = '0.75'
num_robots = 24
num_areas = 8

# value_studied = "mean_timeout"
value_studied = "mean_completion"

area_threshold = 0.082
pickle_file_root = "mean_run_elapsed"
y_lim = 15

if value_studied == "mean_completion":
    area_threshold = 0.1
    pickle_file_root = "mean_run_completion"
    y_lim = 8


def main():
    evaluate_run = False

    results_folder = os.path.join(os.getcwd(), "results_"+walk+"/" + experiment)
    if not os.path.isdir(results_folder):
        print(colored("Error, " + results_folder + " does not exist", 'red'))
    else:
        print(colored("OK, " + results_folder + " exists", 'green'))

    rr = {}
    rb = {}
    br = {}
    bb = {}

    for timeout_folder in natsorted(os.listdir(os.path.join(results_folder))):
        print(colored("Timeout folder:", 'blue'), timeout_folder)

        if timeout_folder.endswith("pickle"):
            continue

        parameters = timeout_folder.split("_")
        for param in parameters:
            if param.startswith("timeout"):
                timeout = int(param.split("#")[-1]) * 10
                # print("\t timeoutR:",timeoutR)

        if timeout == -1:
            print(colored("\tWARNING: wrong timeout folder", 'red'))
            continue


        if os.path.isfile(os.path.join(results_folder, pickle_file_root+"_timeout#"+str(timeout)+"_.pickle")):
            run_memory_mean = pd.read_pickle(
                os.path.join(results_folder,  pickle_file_root+"_timeout#"+str(timeout)+"_.pickle"))
            print(colored(pickle_file_root+"_timeout#"+str(timeout)+"_.pickle already exists for timeout:"+str(timeout), 'green'))
        else:
            # print(colored(
            #     os.path.join(results_folder, pickle_file_root"_timeout#"+str(timeout*10)+"_.pickle"),
            #     'red'))
            # sys.exit()
            for filename in natsorted(os.listdir(os.path.join(results_folder, timeout_folder))):
                filename_seed = filename.split("_")[0]
                # print(filename)
                if filename.endswith("areaLOG_client.tsv"):
                    if not os.path.getsize(os.path.join(results_folder, timeout_folder, filename)) > 0:
                        print(colored("\tWARNING, empty file at:" + filename, 'red'))
                        continue
                    # print('\tfilename: ', filename)
                    df_area_client = pd.read_csv(os.path.join(results_folder, timeout_folder, filename), sep="\t",
                                                 header=None)

                if filename.endswith("areaLOG_server.tsv"):
                    if not os.path.getsize(os.path.join(results_folder, timeout_folder, filename)) > 0:
                        print(colored("\tWARNING, empty file at:" + filename, 'red'))
                        continue
                    # print('\tfilename: ', filename)
                    df_area_server = pd.read_csv(os.path.join(results_folder, timeout_folder, filename), sep="\t",
                                                 header=None)

                # if filename.endswith("taskLOG_client.tsv"):
                #     if not os.path.getsize(os.path.join(results_folder, timeout_folder, filename)) > 0:
                #         print(colored("\tWARNING, empty file at:" + filename, 'red'))
                #         continue
                #     # print('\tfilename: ', filename)
                #     df_task_client = pd.read_csv(os.path.join(results_folder, timeout_folder, filename), sep="\t",
                #                                  header=None)
                #
                # if filename.endswith("taskLOG_server.tsv"):
                #     if not os.path.getsize(os.path.join(results_folder, timeout_folder, filename)) > 0:
                #         print(colored("\tWARNING, empty file at:" + filename, 'red'))
                #         continue
                #     # print('\tfilename: ', filename)
                #     df_task_server = pd.read_csv(os.path.join(results_folder, timeout_folder, filename), sep="\t",
                #                                  header=None)

                if filename.endswith("kiloLOG_client.tsv"):
                    if not os.path.getsize(os.path.join(results_folder, timeout_folder, filename)) > 0:
                        print(colored("\tWARNING, empty file at:" + filename, 'red'))
                        continue
                    # print('\tfilename: ', filename)
                    df_kilo_client = pd.read_csv(os.path.join(results_folder, timeout_folder, filename), sep="\t",
                                                 header=None)

                if filename.endswith("kiloLOG_server.tsv"):
                    if not os.path.getsize(os.path.join(results_folder, timeout_folder, filename)) > 0:
                        print(colored("\tWARNING, empty file at:" + filename, 'red'))
                        continue
                    # print('\tfilename: ', filename, end='\n')
                    df_kilo_server = pd.read_csv(os.path.join(results_folder, timeout_folder, filename), sep="\t",
                                                 header=None)
                    evaluate_run = True

                if evaluate_run:
                    '''Kilo log part'''
                    if len(df_kilo_client.columns) > 145:
                        # print("Cutting null elements in client kilo df")
                        df_kilo_client.drop(df_kilo_client.columns[len(df_kilo_client.columns) - 1], axis=1, inplace=True)

                    if len(df_kilo_server.columns) > 145:
                        # print("Cutting null elements in server kilo df")
                        df_kilo_server.drop(df_kilo_server.columns[len(df_kilo_server.columns) - 1], axis=1, inplace=True)

                    col_kilo_labels = ['time']
                    for i in range(0, len(df_kilo_server.columns) - 1, 6):
                        #     print(i,end=", ")
                        col_kilo_labels += ['id' + str(i // 6), 'state' + str(i // 6), 'posx' + str(i // 6),
                                            'posy' + str(i // 6),
                                            'ori' + str(i // 6), 'same_state' + str(i // 6)]

                    col_kilo_to_drop = []
                    for i in range((len(df_kilo_server.columns) - 1) // 6):
                        #     print(i,end=", ")
                        col_kilo_to_drop += ['same_state' + str(i)]

                    df_kilo_server.columns = col_kilo_labels
                    df_kilo_client.columns = col_kilo_labels
                    df_kilo_server = df_kilo_server.drop(col_kilo_to_drop, axis=1)
                    df_kilo_client = df_kilo_client.drop(col_kilo_to_drop, axis=1)

                    '''Completed task LOG part'''
                    # task_label = ['time', 'id', 'creationTime', 'completitionTime', 'color', 'contained']
                    # df_task_client.columns = task_label

                    '''Area LOG part'''
                    col_area_labels = ['time']
                    for i in range(0, len(df_area_server.columns) - 2, 6):
                        # print(i, end=", ")
                        col_area_labels += ['id' + str(i // 6), 'posx' + str(i // 6), 'posy' + str(i // 6),
                                            'color' + str(i // 6),
                                            'completed' + str(i // 6), 'contained' + str(i // 6)]

                    # Remove last empty col and assign labels to df_area_server
                    if len(df_area_server.columns) > 49:
                        # print("Cutting null elements in area server df")
                        df_area_server.drop(df_area_server.columns[len(df_area_server.columns) - 1], axis=1, inplace=True)
                    df_area_server.columns = col_area_labels

                    # First df_area_client row contains garbage
                    # so is substituted with the second row except for the time,
                    # then remove Nan values in [:,49:]
                    if len(df_area_client.columns) > 49:
                        # print("Cutting null elements in area client df")
                        df_area_client.loc[0, 1:] = df_area_client.loc[1, 1:]
                        df_area_client = df_area_client.drop(np.arange(49, len(df_area_client.columns)), axis=1)
                    df_area_client.columns = col_area_labels

                    area_color_label = []
                    for i in range(num_areas):
                        area_color_label += ["color" + str(i)]
                    #     print("color"+str(i))
                    areas_client_color = df_area_client[area_color_label].iloc[0, :].values
                    areas_server_color = df_area_server[area_color_label].iloc[0, :].values
                    # print(areas_client_color)
                    # print(areas_server_color)

                    area_pos_label = []
                    for i in range(num_areas):
                        area_pos_label += ["posx" + str(i)]
                        area_pos_label += ["posy" + str(i)]
                    areas_pos = df_area_client[area_pos_label].iloc[0, :].values
                    # print(areas_pos)
                    areas_pos = areas_pos.reshape(-1, 2)


                    color_list = ["color" + str(i) for i in range(num_areas)]
                    df_area3_s = df_area_server.iloc[:1, :][color_list]
                    df_area3_c = df_area_client.iloc[:1, :][color_list]
                    for i, idx in enumerate(range(1, len(df_area3_c.columns) * 2, 2)):
                        #     print(i, ' ', idx)
                        df_area3_c.insert(loc=idx, column='other_col' + str(i), value=df_area3_s.iloc[0][i])
                    client = [col for col in df_area3_c.columns if 'color' in col]
                    server = [col for col in df_area3_c.columns if 'other_col' in col]
                    df_area_colors = pd.lreshape(df_area3_c, {'color_client': client, 'color_server': server})
                    area_type = []
                    for area in df_area_colors.values:
                        if area[0] == 0 and area[1] == 0:
                            area_type += ['BB']
                        if area[0] == 0 and area[1] == 1:
                            area_type += ['BR']
                        if area[0] == 1 and area[1] == 0:
                            area_type += ['RB']
                        if area[0] == 1 and area[1] == 1:
                            area_type += ['RR']
                    df_area_colors.insert(loc=2, column='area_type', value=area_type)

                    '''Post process server'''
                    for i, kilo_id in enumerate(np.arange(1, len(df_kilo_server.columns), 5)):
                        # print(colored("kilo_id:" + str((kilo_id - 1) // 5), 'blue'))
                        #     print(df_kilo_client.iloc[:20, kilo_id+2:kilo_id+4].values, end='\n\n')
                        kilo_pos = df_kilo_server.iloc[:, kilo_id + i + 2:kilo_id + i + 4].values
                        #     print(kilo_pos)
                        in_area = np.empty(kilo_pos.shape[0], dtype=int)
                        in_area.fill(-1)
                        for area_idx, area_pos in enumerate(areas_pos):
                            # print(area_idx, ' ', area_pos)
                            dist = np.linalg.norm(kilo_pos - area_pos, axis=1)
                            #     print(dist, end='\n\n')
                            in_area = np.where(dist < area_threshold, df_area_colors.iloc[area_idx][-1][::-1], in_area)
                        #     in_area = np.where(in_area == -1, np.NaN, in_area)
                        #     print(in_area)
                        df_kilo_server.insert(loc=int(kilo_id + i + 2), column='area_type' + str(i), value=in_area)

                    '''Post process client'''
                    for i, kilo_id in enumerate(np.arange(1, len(df_kilo_client.columns), 5)):
                        # print(colored("kilo_id:" + str((kilo_id - 1) // 5), 'blue'))
                        #     print(df_kilo_client.iloc[:20, kilo_id+2:kilo_id+4].values, end='\n\n')
                        kilo_pos = df_kilo_client.iloc[:, kilo_id + i + 2:kilo_id + i + 4].values
                        #     print(kilo_pos)
                        in_area = np.empty(kilo_pos.shape[0], dtype=int)
                        in_area.fill(-1)
                        for area_idx, area_pos in enumerate(areas_pos):
                            #     print(area_idx,' ', area_pos)
                            dist = np.linalg.norm(kilo_pos - area_pos, axis=1)
                            #     print(dist, end='\n\n')
                            in_area = np.where(dist < area_threshold, df_area_colors.iloc[area_idx][-1], in_area)
                        #     in_area = np.where(in_area == -1, np.NaN, in_area)
                        #     print(in_area)
                        df_kilo_client.insert(loc=int(kilo_id + i + 2), column='area_type' + str(i), value=in_area)

                    '''Here finally evaluated in which area the timeout elapses'''
                    kilo_resume = [["state" + str(i), "area_type" + str(i)] for i in range(num_robots)]
                    kilo_resume = np.reshape(kilo_resume, (-1))
                    server_kilo_resume = df_kilo_server.iloc[:][kilo_resume]
                    client_kilo_resume = df_kilo_client.iloc[:][kilo_resume]
                    total_exp_df = client_kilo_resume.join(server_kilo_resume, lsuffix='_c', rsuffix='_s')

                    if value_studied == "mean_timeout":
                        timeout_count = pd.DataFrame(columns=['RR', 'RB', 'BR', 'BB'])
                        for i in range(0, len(total_exp_df.columns), 2):
                            #     print(total_exp_df.iloc[:50,i:i+2])
                            kilo_state = total_exp_df.iloc[:, i:i + 2]
                            kilo_state = kilo_state.replace(2, 3)
                            mask = (kilo_state[kilo_state.columns.values[0]].diff() == 2)
                            #     print(kilo_state[mask])
                            #     print(kilo_state[mask][kilo_state.columns.values[1]].value_counts(), end='\n\n')
                            robot_timeout = kilo_state[mask][kilo_state.columns.values[1]].value_counts().to_frame().T
                            #     robot_timeout = pd.DataFrame(kilo_state[mask][kilo_state.columns.values[1]].value_counts(), columns=['RR, RB,BR,BB'])
                            #     print(robot_timeout)
                            timeout_count = timeout_count.append(robot_timeout)
                            # print(robot_timeout, end='\n\n')
                        timeout_count = timeout_count.fillna(0)
                        single_run_mean = timeout_count.mean(axis=0)

                    else:
                        completed_area_count = pd.DataFrame(columns=['RR', 'RB', 'BR', 'BB'])
                        for i in range(0, len(total_exp_df.columns), 2):
                            #     print(total_exp_df.iloc[:50,i:i+2])
                            kilo_state = total_exp_df.iloc[:, i:i + 2]
                            mask = (kilo_state[kilo_state.columns.values[0]].diff() == -1)
                            # print(kilo_state[mask])
                            # print(kilo_state[mask][kilo_state.columns.values[1]].value_counts(), end='\n\n')
                            robot_completed_area = kilo_state[mask][kilo_state.columns.values[1]].value_counts().to_frame().T
                            #     robot_completed_area = pd.DataFrame(kilo_state[mask][kilo_state.columns.values[1]].value_counts(), columns=['RR, RB,BR,BB'])
                            #     print(robot_completed_area)
                            completed_area_count = completed_area_count.append(robot_completed_area)
                            # print(robot_completed_area, end='\n\n')

                        completed_area_count = completed_area_count.fillna(0)
                        single_run_mean = completed_area_count.mean(axis=0)

                    single_df = single_run_mean.to_frame().T
                    single_df.index = [filename_seed]

                    if os.path.isfile(os.path.join(results_folder, pickle_file_root+"_timeout#"+str(timeout)+"_.pickle")):
                        run_memory_mean = pd.read_pickle(
                            os.path.join(results_folder, pickle_file_root+"_timeout#"+str(timeout)+"_.pickle"))
                        run_memory_mean = run_memory_mean.append(single_df)
                        run_memory_mean.to_pickle(
                            os.path.join(results_folder, pickle_file_root+"_timeout#"+str(timeout)+"_.pickle"))
                        print("Timeout:", timeout, end=", ")
                        print("Appending mean run, file size: ", run_memory_mean.shape)
                    else:
                        print("Timeout:", timeout, end=", ")
                        print("Writing mean run")
                        single_df.to_pickle(os.path.join(results_folder, pickle_file_root+"_timeout#"+str(timeout)+"_.pickle"))
                    evaluate_run = False


        rr[timeout] = run_memory_mean['RR'].values
        rb[timeout] = run_memory_mean['RB'].values
        br[timeout] = run_memory_mean['BR'].values
        bb[timeout] = run_memory_mean['BB'].values

    if value_studied == "mean_timeout":
        figureName = 'meanElapsedTimeout'
    else:
        figureName = 'meanCompletedAreas'

    figureName += '_groupsize'+groupSize+'_'+experiment+'_'+walk
    print("rr", rr)
    boxplots_utils.grouped_4_boxplot(rr, rb, br, bb, y_lim, figureName)




if __name__ == '__main__':
    main()
