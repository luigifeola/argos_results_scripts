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
import warnings

from utils import utils
from utils import from_origin_distribution
from utils import time_stats
from utils import wmsd_heatmaps
from utils import wmsd_time_history


def main():
    folder_experiments = "2020-05-25_simple_collision_avoidance_experiment"
    windowed = False

    '''*********************FLAGS***************************************************************************************
    + If you want WMSD heatmaps set True wmsd_heatmaps_flag 
    + If you want comparison with baseline and arena distribution set True comparison_plots_flag
    + If it is an open space experiment for diffusion evaluation set True open_space_flag
    + Time stats: for convergence time and first passage time stats set True time_stats_flag
    + Pdf plots generation : set True generate_pdf_flag
    + bias_flag True to choose baseline_openspace as baseline folder
    *****************************************************************************************************************'''
    wmsd_heatmaps_flag = True
    comparison_plots_flag = True
    open_space_flag = False
    time_stats_flag = True
    generate_pdf_flag = False
    bias_flag = True

    main_folder = "./results"
    '''Folder baseline'''
    # be careful on this path and especially its sub-folder if you generate new baseline results


    if bias_flag:
        baseline_dir = os.path.join(main_folder, "baseline_openspace")
        if not os.path.exists(baseline_dir):
            print("Baseline:"+baseline_dir+" not an existing directory")
            exit(-1)
        baseline_dir += "/2020-05-28_robots#1_"
    else:
        baseline_dir = os.path.join(main_folder, "baseline")
        if not os.path.exists(baseline_dir):
            print("Baseline:"+baseline_dir+" not an existing directory")
            exit(-1)
        baseline_dir += "/2020-05-27_robots#1_"

    print("Baseline_dir:"+baseline_dir)


    '''Generate folder to store plots and heatmaps'''
    script_dir = os.path.abspath("")
    results_dir = os.path.join(script_dir, "Plots/" + folder_experiments, "")

    wmsd_dir = os.path.join(results_dir, "WMSD", "")

    if windowed:
        result_time_dir = os.path.join(results_dir, "moving_window")
    else:
        result_time_dir = os.path.join(results_dir, "fixed_window")



    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        print("mkdir:"+results_dir)
    else:
        warnings.warn("WARNING: "+os.path.basename(os.path.normpath(results_dir))+" already exists")
    if not os.path.isdir(result_time_dir):
        os.makedirs(result_time_dir)
        print("mkdir:"+result_time_dir)




    bin_edges = np.linspace(0, 0.475, 20)

    if not os.path.isdir(main_folder + '/' + folder_experiments):
        print("folder_experiments is not an existing directory in result folder")
        exit(-1)

    # TODO : fix the necessity of define folder_baseline in the for loop
    # TODO : fix file_title in time_plot_histogram

    '''******************************************************************'''
    '''***************WMSD evaluations***********************************'''
    '''******************************************************************'''
    if comparison_plots_flag:
        distance_heatmap_dir = os.path.join(wmsd_dir, "distance_heatmap", "")
        if not os.path.isdir(distance_heatmap_dir):
            os.makedirs(distance_heatmap_dir)
            print("mkdir:" + distance_heatmap_dir)
        else:
            print("Error: directory already exists")
            exit(-1)

        wmsd_time_history.evaluate_history_WMSD_and_time_diffusion(main_folder, folder_experiments, baseline_dir,
                                                                   windowed, bin_edges, result_time_dir,
                                                                   distance_heatmap_dir)


    '''***************WMSD Heatmaps***************************************'''
    if wmsd_heatmaps_flag:
        heatmap_dir = os.path.join(result_time_dir, "Heatmap")
        if not os.path.isdir(heatmap_dir):
            os.makedirs(heatmap_dir)
            print("mkdir:"+heatmap_dir)
        else:
            print("Error: directory already exists")
            exit(-1)
        wmsd_heatmaps.evaluate_WMSD_heatmap(main_folder, folder_experiments, windowed, heatmap_dir)

    '''******************************************************************'''
    '''***************Powerlaw for openspace experiments*****************'''
    '''******************************************************************'''
    if open_space_flag:
        powerlaw_dir = os.path.join(results_dir, "origin_distance_distribution", "")
        if not os.path.isdir(powerlaw_dir):
            os.makedirs(powerlaw_dir)
            print("mkdir:" + powerlaw_dir)
        else:
            print("Error: directory already exists")
            exit(-1)

        from_origin_distribution.distance_from_origin_distribution(main_folder, folder_experiments, powerlaw_dir)

    '''********************************************************************'''
    '''******************Convergence and FPT evaluations*******************'''
    '''********************************************************************'''
    if time_stats_flag:
        weibull_dir = os.path.join(results_dir, "Weibull")
        conv_time_dir = os.path.join(weibull_dir, 'convergence_time')
        ftp_dir = os.path.join(weibull_dir, 'first_passage_time')

        if not os.path.isdir(weibull_dir):
            os.makedirs(weibull_dir)
            print("mkdir:" + weibull_dir)
        else:
            print("Error: directory already exists")
            exit(-1)
        if not os.path.isdir(conv_time_dir):
            os.makedirs(conv_time_dir)
            print("mkdir:" + conv_time_dir)
        if not os.path.isdir(ftp_dir):
            os.makedirs(ftp_dir)
            print("mkdir:" + ftp_dir)

        ''' When conv_time_estimation==False -> fpt estimation'''
        convergence_time_estimation = False
        bound_is = 75000
        folder_experiments = os.path.join(main_folder, folder_experiments)
        time_stats.evaluate_time_stats(folder_experiments, conv_time_dir, ftp_dir, convergence_time_estimation, bound_is)

    if generate_pdf_flag:
        utils.generate_pdf(results_dir)

if __name__ == '__main__':
    main()