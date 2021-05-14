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

area_threshold = 0.1
num_robots = 24
num_areas = 8


def main():
    evaluate_run = False

    results_folder = os.path.join(os.getcwd(), "results_"+walk+"/" + experiment)
    if not os.path.isdir(results_folder):
        print(colored("Error, " + results_folder + " does not exist", 'red'))
    else:
        print(colored("OK, " + results_folder + " exists", 'green'))


    for timeout_folder in natsorted(os.listdir(os.path.join(results_folder))):
        if timeout_folder.endswith("pickle"):
            continue

        print(colored("Timeout folder:", 'blue'), timeout_folder)
        df_kilo_timeout = pd.DataFrame()

        timeout = -1
        parameters = timeout_folder.split("_")
        for param in parameters:
            if param.startswith("timeout"):
                timeout = int(param.split("#")[-1]) * 10
                # print("\t timeoutR:",timeoutR)

        if timeout == -1:
            print(colored("\tWARNING: wrong timeout folder", 'red'))
            continue

        if os.path.isfile(os.path.join(results_folder, timeout_folder, "kiloLOG_timeout#" + str(timeout) + "_.pickle")):
            print("Already exists ",
                  os.path.join(results_folder, timeout_folder, "kiloLOG_timeout#" + str(timeout) + "_.pickle"))

        else:
            # print(colored(
            #     os.path.join(results_folder, pickle_file_root"_timeout#"+str(timeout*10)+"_.pickle"),
            #     'red'))
            # sys.exit()
            for filename in natsorted(os.listdir(os.path.join(results_folder, timeout_folder))):
                filename_seed = filename.split("_")[0].split("#")[-1]
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
                    print(colored("\tEvaluating run:" + filename_seed, 'blue'))


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
                    for i_c, kilo_id in enumerate(np.arange(1, len(df_kilo_server.columns), 5)):
                        # print(colored("kilo_id:" + str((kilo_id - 1) // 5), 'blue'))
                        #     print(df_kilo_client.iloc[:20, kilo_id+2:kilo_id+4].values, end='\n\n')
                        kilo_pos = df_kilo_server.iloc[:, kilo_id + i_c + 2:kilo_id + i_c + 4].values
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
                        df_kilo_server.insert(loc=int(kilo_id + i_c + 2), column='area_type' + str(i_c), value=in_area)

                    '''Post process client'''
                    for i_s, kilo_id in enumerate(np.arange(1, len(df_kilo_client.columns), 5)):
                        # print(colored("kilo_id:" + str((kilo_id - 1) // 5), 'blue'))
                        #     print(df_kilo_client.iloc[:20, kilo_id+2:kilo_id+4].values, end='\n\n')
                        kilo_pos = df_kilo_client.iloc[:, kilo_id + i_s + 2:kilo_id + i_s + 4].values
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
                        df_kilo_client.insert(loc=int(kilo_id + i_s + 2), column='area_type' + str(i_s), value=in_area)

                    df_kilo_single_run = df_kilo_client.join(df_kilo_server, lsuffix='_c', rsuffix='_s')
                    df_kilo_single_run = df_kilo_single_run.set_index(df_kilo_single_run.index.astype(str) + '_' + filename_seed)

                    df_kilo_timeout = df_kilo_timeout.append(df_kilo_single_run)


                    evaluate_run = False

            '''Save pickle file'''
            df_kilo_timeout.to_pickle(os.path.join(results_folder, timeout_folder, "kiloLOG_timeout#"+str(timeout)+"_.pickle"))
            print("Saving at: ", os.path.join(results_folder, timeout_folder, "kiloLOG_timeout#"+str(timeout)+"_.pickle"))
            print("Changing dir")






if __name__ == '__main__':
    main()
