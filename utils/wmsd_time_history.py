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

from utils import config

### WMSD in time & "Hystogram2d"
def evaluate_history_WMSD_and_time_diffusion(main_folder, folder_experiments, baseline_dir, windowed, b_edges,
                                             result_time_dir, distance_heatmap_dir):
    for dirName, subdirList, fileList in os.walk(main_folder + '/' + folder_experiments):

        # print(dirName)
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
        if (num_robots == "-1" or rho == -1.0 or alpha == -1):
            continue

        # print("dirName: ", dirName)
        runs = len([f for f in fileList if
                    (os.path.isfile(os.path.join(dirName, f)) and f.endswith('position.tsv'))])
        # print("runs: ", runs)

        rho_str = str(rho)
        alpha_str = str(alpha)
        # print("rho", rho_str)
        # print("alpha", alpha_str)

        total_experiment_wmsd = []
        baseline_experiment_wmsd = []

        #     print(alpha_str)

        folder_baseline = baseline_dir+"alpha#%s_rho#%s_baseline_1800" % (alpha_str, rho_str)
        # if not os.path.isdir(main_folder + '/' + folder_baseline):
        #     print("folder_baseline is not an existing directory")
        #     exit(-1)

        number_of_experiments = 0
        df_experiment = pd.DataFrame()
        df_baseline = pd.DataFrame()

        #         print("W_size=", window_size)
        [number_of_experiments, df_experiment] = utils.load_pd_positions(dirName, "experiment")
        [_, df_baseline] = utils.load_pd_positions(folder_baseline, "baseline")

        #     print(number_of_experiments)
        positions_concatenated = df_experiment.values[:, 1:]
        [num_robot, num_times] = positions_concatenated.shape
        positions_concatenated = np.array([x.split(',') for x in positions_concatenated.ravel()], dtype=float)
        positions_concatenated = positions_concatenated.reshape(num_robot, num_times, 2)
        #         print(positions_concatenated.shape)

        baseline_concatenated = df_baseline.values[:, 1:]
        [num_robot, num_times] = baseline_concatenated.shape
        baseline_concatenated = np.array([x.split(',') for x in baseline_concatenated.ravel()], dtype=float)
        baseline_concatenated = baseline_concatenated.reshape(num_robot, num_times, 2)

        for window_size in range(1, 10):
            w_displacement_array = np.array([])
            base_w_displacement_array = np.array([])

            if (windowed):
                base_win_disp = utils.window_displacement(baseline_concatenated, window_size)
                win_disp = utils.window_displacement(positions_concatenated, window_size)
            else:
                win_disp = utils.fixed_window_displacement(positions_concatenated, window_size)
                base_win_disp = utils.fixed_window_displacement(baseline_concatenated, window_size)
            w_displacement_array = np.vstack(
                [w_displacement_array, win_disp]) if w_displacement_array.size else win_disp
            base_w_displacement_array = np.vstack(
                [base_w_displacement_array, base_win_disp]) if base_w_displacement_array.size else base_win_disp
            mean_wmsd = win_disp.mean()

            total_experiment_wmsd.append(w_displacement_array)
            baseline_experiment_wmsd.append(base_w_displacement_array)

        utils.plot_both_wmsd(windowed, baseline_experiment_wmsd, total_experiment_wmsd, alpha_str, rho_str, num_robots,
                       result_time_dir)

        # distance_heatmap
        distances = utils.distance_from_the_origin(positions_concatenated)
        occurrences = utils.get_occurrences(distances, b_edges, runs)

        if not config.open_space_flag:
            utils.time_plot_histogram(occurrences.T, b_edges[1:], alpha_str, rho_str, num_robots,
                                distance_heatmap_dir)
