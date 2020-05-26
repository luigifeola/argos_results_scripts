import sys
import os
from utils import utils
import numpy as np

def main():

    main_folder = "./results"
    folder_experiments = "2020-04-21_bias_experiment"
    windowed = False

    from datetime import date
    today = date.today()

    '''Generate folder to store plots and heatmaps'''
    script_dir = os.path.abspath('')
    results_dir = os.path.join(script_dir, 'Plots/'+str(today)+'/WMSD')

    distance_heatmap_dir = os.path.join(results_dir, 'distance_heatmap')

    if (windowed):
        result_time_dir = os.path.join(results_dir, 'moving_window')
    else:
        result_time_dir = os.path.join(results_dir, 'fixed_window')

    heatmap_dir = os.path.join(result_time_dir, 'Heatmap')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if not os.path.isdir(result_time_dir):
        os.makedirs(result_time_dir)
    if not os.path.isdir(heatmap_dir):
        os.makedirs(heatmap_dir)
    if not os.path.isdir(distance_heatmap_dir):
        os.makedirs(distance_heatmap_dir)


    bin_edges = np.linspace(0, 0.475,20)

    if not os.path.isdir(main_folder+'/'+folder_experiments):
        print("folder_experiment is not an existing directory")
        exit(-1)


    #TODO : fix the necessity of define folder_baseline in the for loop
    #TODO : fix file_title in time_plot_histogram
    #TODO : pass also the folder baseline

    # utils.evaluate_history_WMSD_and_time_diffusion(main_folder, folder_experiments, windowed, bin_edges, result_time_dir, distance_heatmap_dir)
    # utils.evaluate_WMSD_heatmap(main_folder, folder_experiments, windowed, heatmap_dir)

    utils.origin_distance_distribution(main_folder, folder_experiments, heatmap_dir)

if __name__ == '__main__':
    main()

# path = "/home/luigi/Documents/scripts/test_scripts/results/results_2020-02-21_rob_20/2020-02-21_robots#20_alpha#1.2_rho#0.0_experiment_1800/experiment.pkl"
# df = pd.read_pickle(path)
