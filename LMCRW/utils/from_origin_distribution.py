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
import powerlaw
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from utils import utils


def distance_from_origin_distribution(main_folder, folder_experiments, powerlaw_dir):
    Ncolors = 200
    colormap = plt.cm.viridis  # LinearSegmentedColormap
    #     print("colormap.N", colormap.N)
    Ncolors = min(colormap.N, Ncolors)
    mapcolors = [colormap(int(x * colormap.N / Ncolors)) for x in range(Ncolors)]
    #     print(len(mapcolors))

    for dirName, subdirList, fileList in os.walk(main_folder + '/' + folder_experiments):

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

        if (num_robots == "-1" or rho == -1.0 or alpha == -1):
            continue

        rho_str = str(rho)
        alpha_str = str(alpha)
        #     print("rho", rho_str)
        #     print("alpha", alpha_str)
        #     print(dirName)

        df_experiment = pd.DataFrame()

        [_, df_experiment] = utils.load_pd_positions(dirName, "experiment")

        #     print(number_of_experiments)
        positions_concatenated = df_experiment.values[:, 1:]
        [num_robot, num_times] = positions_concatenated.shape
        positions_concatenated = np.array([x.split(',') for x in positions_concatenated.ravel()], dtype=float)
        positions_concatenated = positions_concatenated.reshape(num_robot, num_times, 2)
        # print("positions_concatenated.shape", positions_concatenated.shape)

        distances = utils.distance_from_the_origin(positions_concatenated)
        print("distances.shape", distances.shape)

        fig = plt.figure(figsize=(10, 5), dpi=160)
        plt.xlim((0.001, 100))
        plt.ylim((0.0001, 100))
        for i, d in enumerate(distances):
            fit = powerlaw.Fit(d, xmin=0.00001)
            fit.plot_pdf(linewidth=2, color=mapcolors[i])
        plt.ylabel('p(x)')
        plt.xlabel('distance from origin')
        plt.title("origin distance distribution with %s robots, alpha=%s, rho=%s" % (num_robots, alpha, rho))
        file_name = "powerlaw_%s_rho_%s_alpha_%s.png" % (num_robots, rho, alpha)
        plt.savefig(powerlaw_dir + '/' + file_name)
        plt.close(fig)

