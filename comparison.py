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

Ncolors = 9
colormap = plt.cm.viridis# LinearSegmentedColormap
Ncolors = min(colormap.N,Ncolors)
mapcolors = [colormap(int(x*colormap.N/Ncolors)) for x in range(Ncolors)]

rho = "0.0"
alpha = "1.2"
robot = 0
results_dir = "./comparison/rho_%s_alpha_%s/" % (rho, alpha)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

for seed_baseline in range(1, 501):
    for seed in range(1, 51):

        folder_baseline_seed = "./results/2020-04-21_baseline/2020-04-21_robots#1_alpha#" + alpha + "_rho#" + rho + "_baseline_1800/seed#%d_position.tsv" % seed_baseline
        # print(folder_baseline_seed1)
        robot_base_seed = pd.read_csv(folder_baseline_seed, sep="\t").values[0, 1:]

        robot_base = np.array([x.split(',') for x in robot_base_seed.ravel()], dtype=float)
        robot_base_x = robot_base[:, 0]
        robot_base_y = robot_base[:, 1]

        dirName_seed1 = "./results/2020-04-21_bias_experiment/2020-04-21_robots#20_alpha#" + alpha + "_rho#" + rho + "_sim_1800/seed#%d_position.tsv" % seed
        robot_seed = pd.read_csv(dirName_seed1, sep="\t").values[robot, 1:]
        robot_pos = np.array([x.split(',') for x in robot_seed.ravel()], dtype=float)
        robot_x = robot_pos[:, 0]
        robot_y = robot_pos[:, 1]






        fig = plt.figure(figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
        # plt.xlim((-10, 10))
        # plt.ylim((-10, 10))
        plt.plot(robot_base_x, robot_base_y, marker='.', label="baseline", color=mapcolors[3])
        label = "robot%d" % robot
        plt.plot(robot_x, robot_y, marker='.', label=label, color=mapcolors[0])

        figName = "seedbaseline_%d_robot_%d_seed_%d" % (seed_baseline, robot, seed)
        plt.title(figName)
        fig.legend(loc=2)

        plt.savefig(results_dir + figName + ".png")
        plt.close(fig)
