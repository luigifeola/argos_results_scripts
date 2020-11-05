import os

from utils import utils
from utils import config
from utils import from_origin_distribution
from utils import time_stats
from utils import wmsd_heatmaps
from utils import wmsd_time_history
from utils import average_connection_degree
from utils import cluster_estimation
from termcolor import colored


# TODO : fix the necessity of define folder_baseline in the for loop
# TODO : fix file_title in time_plot_histogram
# TODO : fix mkdir directories just if you need it and not before
# TODO : add axis label to the plots (especially to the heatmaps)

def main():
    main_folder = os.path.join(os.getcwd(), "results")

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
    results_dir = os.path.join(main_folder, "Plots/" + config.folder_experiments, "")

    # Check if experiment result folder plots exists, else raise a warning
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        print("mkdir:" + results_dir)
    else:
        print(colored('WARNING: already exists directory: ' + results_dir, 'green'))


    '''******************************************************************'''
    '''***************WMSD evaluations***********************************'''
    '''******************************************************************'''
    if config.comparison_plots_flag or config.density_maps_flag:

        if config.msd_type == 'windowed':
            result_time_dir = os.path.join(results_dir, "moving_window_displacement")
        elif config.msd_type == 'fixed':
            result_time_dir = os.path.join(results_dir, "fixed_window_displacement")
        elif config.msd_type == 'time_msd':
            result_time_dir = os.path.join(results_dir, "time_msd_displacement")
        else:
            print(colored('Error, bad msd_type', 'red'))

        heatmap_dir = os.path.join(result_time_dir, "Heatmap")
        density_maps_dir = os.path.join(results_dir, "density_maps", "")

        if config.comparison_plots_flag:
            if not os.path.isdir(result_time_dir):
                os.makedirs(result_time_dir)
                print("mkdir:" + result_time_dir)
            else:
                print(colored('ERROR: already exists directory: ' + result_time_dir, 'red'))
                exit(-1)

            '''***************WMSD Heatmaps***************************************'''
            if config.wmsd_heatmaps_flag:
                if not os.path.isdir(heatmap_dir):
                    os.makedirs(heatmap_dir)
                    print("mkdir:" + heatmap_dir)
                    wmsd_heatmaps.evaluate_WMSD_heatmap(main_folder, config.folder_experiments, baseline_dir,
                                                        config.msd_type,
                                                        heatmap_dir)
                else:
                    print(colored('ERROR: already exists directory: ' + heatmap_dir, 'red'))
                    exit(-1)

        if config.density_maps_flag:
            if not os.path.isdir(density_maps_dir):
                os.makedirs(density_maps_dir)
                print("mkdir:" + density_maps_dir)
            else:
                print(colored('ERROR: already exists directory: ' + density_maps_dir, 'red'))
                exit(-1)

        wmsd_time_history.evaluate_history_WMSD_and_time_diffusion(main_folder, config.folder_experiments,
                                                                   baseline_dir, config.msd_type, config.bin_edges,
                                                                   result_time_dir, density_maps_dir)


    '''******************************************************************'''
    '''*************average connection degree evaluations****************'''
    '''******************************************************************'''
    if config.connection_degree_flag:
        avg_connection_degree_dir = os.path.join(results_dir, "average_connection_degree", "")
        if not os.path.isdir(avg_connection_degree_dir):
            os.makedirs(avg_connection_degree_dir)
            print("mkdir:" + avg_connection_degree_dir)

            folder_experiment = os.path.join(main_folder, config.folder_experiments)
            average_connection_degree.avg_connection_plot_different_population_sizes(folder_experiment,
                                                                                     avg_connection_degree_dir)
            average_connection_degree.avg_connection_degree_heatmap(folder_experiment, avg_connection_degree_dir)

        else:
            print(colored('ERROR: already exists directory: ' + avg_connection_degree_dir, 'red'))
            exit(-1)


    '''******************************************************************'''
    '''*************cluster estimation***********************************'''
    '''******************************************************************'''
    if config.cluster_estimation_flag:
        cluster_dir = os.path.join(results_dir, "cluster_dir", "")
        if not os.path.isdir(cluster_dir):
            os.makedirs(cluster_dir)
            print("mkdir:" + cluster_dir)

            folder_experiment = os.path.join(main_folder, config.folder_experiments)
            cluster_estimation.cluster_estimation_study(folder_experiment, cluster_dir)

        else:
            print(colored('ERROR: cluster_dir already exists at: ' + cluster_dir, 'red'))
            exit(-1)


    '''******************************************************************'''
    '''***************Powerlaw for open-space experiments*****************'''
    '''******************************************************************'''
    if config.open_space_flag:
        powerlaw_dir = os.path.join(results_dir, "open_space_distribution", "")
        if not os.path.isdir(powerlaw_dir):
            os.makedirs(powerlaw_dir)
            print("mkdir:" + powerlaw_dir)
            from_origin_distribution.distance_from_origin_distribution(main_folder, config.folder_experiments,
                                                                       powerlaw_dir)

        else:
            print(colored('ERROR: already exists directory: ' + powerlaw_dir, 'red'))
            exit(-1)


    '''********************************************************************'''
    '''******************Convergence and FPT evaluations*******************'''
    '''********************************************************************'''
    bound_is = 75000
    folder_experiment = os.path.join(main_folder, config.folder_experiments)

    if config.time_stats_flag:
        time_stats_dir = os.path.join(results_dir, "time_stats")
        conv_time_dir = os.path.join(time_stats_dir, 'convergence_time')
        ftp_dir = os.path.join(time_stats_dir, 'first_passage_time')

        if not os.path.isdir(time_stats_dir):
            os.makedirs(time_stats_dir)
            print("mkdir:" + time_stats_dir)

        else:
            print(colored('ERROR: already exists directory: ' + time_stats_dir, 'red'))
            exit(-1)

        if not os.path.isdir(conv_time_dir):
            os.makedirs(conv_time_dir)
            print("mkdir:" + conv_time_dir)
            ''' When conv_time_estimation==False -> fpt estimation'''
            convergence_time_estimation = True
            time_stats.evaluate_time_stats(folder_experiment, conv_time_dir, ftp_dir, convergence_time_estimation,
                                           bound_is)

        else:
            print(colored('WARNING: already exists directory: ' + conv_time_dir, 'green'))

        if not os.path.isdir(ftp_dir):
            os.makedirs(ftp_dir)
            print("mkdir:" + ftp_dir)
            convergence_time_estimation = False
            time_stats.evaluate_time_stats(folder_experiment, conv_time_dir, ftp_dir, convergence_time_estimation,
                                           bound_is)

        else:
            print(colored('ERROR: already exists directory: ' + ftp_dir, 'red'))
            exit(-1)


    '''********************************************************************'''
    '''******************Generate PDF with all results*********************'''
    '''********************************************************************'''

    if config.generate_pdf_flag:
        print("Generating pdf Plots in folder: ", results_dir)
        utils.generate_pdf(results_dir)


if __name__ == '__main__':
    main()
