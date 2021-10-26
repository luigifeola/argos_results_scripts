import numpy as np

# folder_experiments = "bias_experiment_100_runs"
folder_experiments = "bouncing_angle_100_runs"
# folder_experiments = "simple_experiment_100_runs"
# folder_experiments = "random_angle_100_runs"

population_size = 6  # len(10 20 50 100) robots
alpha_array = [1.2, 1.4, 1.6, 1.8, 2.0]
rho_array = [0.0, 0.3, 0.6, 0.9]

'''*********************FLAGS***************************************************************************************
+ Specify the MSD type you want, msd_type possible values: fixed, windowed, time_msd. "windowed" for WMSD, fixed 
    for fixed window MSD
+ If you want WMSD heatmaps set True wmsd_heatmaps_flag 
+ If you want comparison with baseline and arena distribution set True comparison_plots_flag
+ If it is an open space experiment for diffusion evaluation set True open_space_flag 
+ If you want Density Maps in closed space scenario set density_maps_flag to True
+ Time stats: for convergence time and first passage time stats set True time_stats_flag
+ baseline_openspace_flag : set True to choose baseline_openspace as baseline folder
+ connection_degree_flag : set True to evaluate the average connection degree among different population sizes and 
                           to evaluate the average connection degree distributed over different circular section 
                           positions 
+ generate_pdf_flag : set True to generate plots in a single pdf file 
*****************************************************************************************************************'''

msd_type = "fixed"
wmsd_heatmaps_flag = True
comparison_plots_flag = True

distance_parameter = 1.1
kilo_diameter = 0.033
cluster_estimation_flag = False

# TODO : take the following flag directly from the folder name
open_space_flag = False
density_maps_flag = not open_space_flag

time_stats_flag = True
baseline_openspace_flag = False  # this is set almost always to False
connection_degree_flag = True
generate_pdf_flag = False

# TODO : fix the flags if openspace origin distance distribution not needed
if open_space_flag:
    bin_edges = np.linspace(0, 1, 20)  # 20 bins from 0 to 1 meter
else:
    bin_edges = np.linspace(0, 0.45, 20)
