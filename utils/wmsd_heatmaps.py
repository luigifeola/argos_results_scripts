from utils import utils
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
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

def evaluate_WMSD_heatmap(main_folder, folder_experiments, baseline_dir, windowed, heatmap_dir):
    for window_size in range(1,10):

        total_dict=dict()
        number_dict=dict()

        for dirName, subdirList, fileList in os.walk(main_folder+'/'+folder_experiments):

            num_robots = "-1"
            rho = -1.0
            alpha = -1.0
            elements=dirName.split("_")
            for e in elements:
                if e.startswith("robots"):
                    num_robots=e.split("#")[-1]
                    if(num_robots not in total_dict):
                        total_dict[num_robots]=dict()
                        number_dict[num_robots]=dict()



                if(e.startswith("rho")):
                    rho=float(e.split("#")[-1])
                if(e.startswith("alpha")):
                    alpha=float(e.split("#")[-1])

        #     print(str(count) + " : " + dirName)
            if(num_robots == "-1" or rho == -1.0 or alpha == -1):
                continue


            rho_str=str(rho)
            alpha_str=str(alpha)
        #     print("rho", rho_str)
        #     print("alpha", alpha_str)
        #     print(dirName)
            if(rho_str not in total_dict[num_robots]):
                total_dict[num_robots][rho_str]=dict()
                number_dict[num_robots][rho_str]=dict()
        #         print(total_dict)


            total_experiment_wmsd = []
            baseline_experiment_wmsd = []

            # folder_baseline = "baseline_2020-02-14/2020-02-14_robots#1_alpha#%s_rho#%s_baseline_1800" %(alpha_str, rho_str)
            folder_baseline = baseline_dir + "alpha#%s_rho#%s_baseline_1800" % (alpha_str, rho_str)

            number_of_experiments = 0
            df_experiment = pd.DataFrame()
            df_baseline = pd.DataFrame()

        #         print("W_size=", window_size)
            [number_of_experiments, df_experiment] = utils.load_pd_positions(dirName, "experiment")
            [_, df_baseline] = utils.load_pd_positions(folder_baseline, "baseline")


        #     print(number_of_experiments)
            positions_concatenated = df_experiment.values[:,1:]
            [num_robot, num_times] = positions_concatenated.shape
            positions_concatenated = np.array([x.split(',') for x in positions_concatenated.ravel()],dtype=float)
            positions_concatenated = positions_concatenated.reshape(num_robot,num_times,2)


            baseline_concatenated = df_baseline.values[:,1:]
            [num_robot, num_times] = baseline_concatenated.shape
            baseline_concatenated = np.array([x.split(',') for x in baseline_concatenated.ravel()],dtype=float)
            baseline_concatenated = baseline_concatenated.reshape(num_robot,num_times,2)





            w_displacement_array = np.array([])
            base_w_displacement_array = np.array([])

            if(windowed):
                base_win_disp = utils.window_displacement(baseline_concatenated, window_size)
                win_disp = utils.window_displacement(positions_concatenated, window_size)
            else:
                # win_disp = utils.fixed_window_displacement(positions_concatenated, window_size)
                # base_win_disp = utils.fixed_window_displacement(baseline_concatenated, window_size)
                win_disp = utils.time_mean_square_displacement(positions_concatenated)
                base_win_disp = utils.time_mean_square_displacement(baseline_concatenated)
            w_displacement_array = np.vstack([w_displacement_array, win_disp]) if w_displacement_array.size else win_disp
            base_w_displacement_array = np.vstack([base_w_displacement_array, base_win_disp]) if base_w_displacement_array.size else base_win_disp
            mean_wmsd = win_disp.mean()

            total_dict[num_robots][rho_str][alpha_str] = mean_wmsd
            number_dict[num_robots][rho_str][alpha_str] = number_of_experiments
            total_experiment_wmsd.append(w_displacement_array)
            baseline_experiment_wmsd.append(base_w_displacement_array)

    #         print(heatmap_dir)
#             print(total_dict)
            total_dict = utils.sort_nested_dict(total_dict)
            utils.plot_heatmap(total_dict, window_size, heatmap_dir)
