import os
import warnings

from utils import utils
from utils import config
from utils import from_origin_distribution
from utils import time_stats
from utils import wmsd_heatmaps
from utils import wmsd_time_history
from utils import average_connection_degree

# TODO : fix the necessity of define folder_baseline in the for loop
# TODO : fix file_title in time_plot_histogram
# TODO : fix mkdir directories just if you need it and not before
# TODO : add axis label to the plots (especially to the heatmaps)

def main():
    # config.folder_experiments = "2020-05-25_simple_collision_avoidance_experiment"


    main_folder = os.path.join(os.getcwd(), "results")

    # For debug
    # main_folder = os.path.join(os.getcwd(), "test")
    # print("Main path:", main_folder)

    # Check if experiment result folder exists
    if not os.path.isdir(main_folder + '/' + config.folder_experiments):
        print("folder_experiments is not an existing directory in result folder", main_folder + '/'
              + config.folder_experiments)
        exit(-1)

    # Choose which baseline you need
    if config.baseline_openspace_flag:
        baseline_dir = os.path.join(main_folder, "baseline_openspace")
        if not os.path.exists(baseline_dir):
            print("Baseline:" + baseline_dir + " not an existing directory")
            exit(-1)
    else:
        baseline_dir = os.path.join(main_folder, "baseline")
        if not os.path.exists(baseline_dir):
            print("Baseline:" + baseline_dir + " not an existing directory")
            exit(-1)

    baseline_dir += "/2020-05-30_robots#1_"
    print("Baseline_dir:" + baseline_dir)


    '''**************************Generate folder to store plots and heatmaps*************************************'''
    script_dir = os.path.abspath("")
    # results_dir = os.path.join(script_dir, "Plots/" + config.folder_experiments, "")

    # For debug
    results_dir = os.path.join(main_folder, "Plots/" + config.folder_experiments, "")

    # Check if experiment result folder plots exists, else raise a warning
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        print("mkdir:" + results_dir)
    else:
        warnings.warn("WARNING: results_dir already exists")

    '''******************************************************************'''
    '''***************WMSD evaluations***********************************'''
    '''******************************************************************'''
    if config.comparison_plots_flag or config.wmsd_heatmaps_flag:
        if config.windowed:
            result_time_dir = os.path.join(results_dir, "moving_window_displacement")
        else:
            result_time_dir = os.path.join(results_dir, "fixed_window_displacement")

        if not os.path.isdir(result_time_dir):
            os.makedirs(result_time_dir)
            print("mkdir:" + result_time_dir)

    '''***************WMSD comparison plots***********************************'''
    if config.comparison_plots_flag:
        density_maps_dir = os.path.join(results_dir, "density_maps", "")
        if not os.path.isdir(density_maps_dir):
            os.makedirs(density_maps_dir)
            print("mkdir:" + density_maps_dir)
        else:
            print("Error: directory already exists")
            exit(-1)

        wmsd_time_history.evaluate_history_WMSD_and_time_diffusion(main_folder, config.folder_experiments,
                                                                   baseline_dir, config.windowed, config.bin_edges,
                                                                   result_time_dir, density_maps_dir)

    '''***************WMSD Heatmaps***************************************'''
    if config.wmsd_heatmaps_flag:
        heatmap_dir = os.path.join(result_time_dir, "Heatmap")
        if not os.path.isdir(heatmap_dir):
            os.makedirs(heatmap_dir)
            print("mkdir:" + heatmap_dir)
        else:
            print("Error: directory already exists")
            exit(-1)
        wmsd_heatmaps.evaluate_WMSD_heatmap(main_folder, config.folder_experiments, baseline_dir, config.windowed,
                                            heatmap_dir)


    '''******************************************************************'''
    '''*************average connection degree evaluations****************'''
    '''******************************************************************'''
    if config.connection_degree_flag:
        avg_connection_degree_dir = os.path.join(results_dir, "average_connection_degree", "")
        if not os.path.isdir(avg_connection_degree_dir):
            os.makedirs(avg_connection_degree_dir)
            print("mkdir:" + avg_connection_degree_dir)
        else:
            print("Error: directory "+avg_connection_degree_dir+" already exists")
            exit(-1)

        folder_experiment = os.path.join(main_folder, config.folder_experiments)
        average_connection_degree.avg_connection_plot_different_population_sizes(folder_experiment, avg_connection_degree_dir)
        average_connection_degree.avg_connection_degree_heatmap(folder_experiment, avg_connection_degree_dir)


    '''******************************************************************'''
    '''***************Powerlaw for openspace experiments*****************'''
    '''******************************************************************'''
    if config.open_space_flag:
        powerlaw_dir = os.path.join(results_dir, "open_space_distribution", "")
        if not os.path.isdir(powerlaw_dir):
            os.makedirs(powerlaw_dir)
            print("mkdir:" + powerlaw_dir)
        else:
            print("Error: directory already exists")
            exit(-1)

        from_origin_distribution.distance_from_origin_distribution(main_folder, config.folder_experiments, powerlaw_dir)

    '''********************************************************************'''
    '''******************Convergence and FPT evaluations*******************'''
    '''********************************************************************'''
    if config.time_stats_flag:
        time_stats_dir = os.path.join(results_dir, "time_stats")
        conv_time_dir = os.path.join(time_stats_dir, 'convergence_time')
        ftp_dir = os.path.join(time_stats_dir, 'first_passage_time')

        if not os.path.isdir(time_stats_dir):
            os.makedirs(time_stats_dir)
            print("mkdir:" + time_stats_dir)
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
        convergence_time_estimation = True
        bound_is = 75000
        folder_experiment = os.path.join(main_folder, config.folder_experiments)
        time_stats.evaluate_time_stats(folder_experiment, conv_time_dir, ftp_dir, convergence_time_estimation,
                                       bound_is)
        convergence_time_estimation = False
        time_stats.evaluate_time_stats(folder_experiment, conv_time_dir, ftp_dir, convergence_time_estimation,
                                       bound_is)

    if config.generate_pdf_flag:
        print("Generating pdf Plots in folder: ", results_dir)
        utils.generate_pdf(results_dir)


if __name__ == '__main__':
    main()
