import numpy as np


folder_experiments = "simple_experiment"


population_size = 5
alpha_array = [1.2, 1.6, 2.0]
rho_array = [0.0, 0.3, 0.6, 0.9]

windowed = False    # True for overlapping WMSD, False for non-overlapping

'''*********************FLAGS***************************************************************************************
+ If you want WMSD heatmaps set True wmsd_heatmaps_flag 
+ If you want comparison with baseline and arena distribution set True comparison_plots_flag
+ If it is an open space experiment for diffusion evaluation set True open_space_flag
+ Time stats: for convergence time and first passage time stats set True time_stats_flag
+ baseline_openspace_flag : set True to choose baseline_openspace as baseline folder
+ connection_degree_flag : set True to evaluate the average connection degree among different population sizes and 
                           to evaluate the average connection degree distribuited over different circular section 
                           positions 
+ generate_pdf_flag : set True to generate plots in a single pdf file 
*****************************************************************************************************************'''
wmsd_heatmaps_flag = True
comparison_plots_flag = True
open_space_flag = False
time_stats_flag = True
baseline_openspace_flag = False  # this is set almost always to False
connection_degree_flag = True
generate_pdf_flag = True

# TODO : fix average connection degree bin_edge for open space experiment
if open_space_flag:
    # bin_edges = np.linspace(0, 1, 20)
    bin_edges = np.linspace(0, 0.45, 20)
else:
    bin_edges = np.linspace(0, 0.45, 20)
