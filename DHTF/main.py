import os
from utils import utils
from utils import config
import pandas as pd
from termcolor import colored
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import csv


def main():
    main_folder = os.path.join(os.getcwd(), "results")
    dirPath = main_folder + '/' + config.folder_experiments

    # # Check if experiment result folder exists
    # if not os.path.isdir(main_folder + '/' + config.folder_experiments):
    #     print("folder_experiments is not an existing directory in result folder", main_folder + '/'
    #           + config.folder_experiments)
    #     exit(-1)
    #
    # '''**************************Generate folder to store plots*************************************'''
    # results_dir = os.path.join(main_folder, "Plots/" + config.folder_experiments, "")
    #
    # # Check if experiment result folder plots exists, else raise a warning
    # if not os.path.isdir(results_dir):
    #     os.makedirs(results_dir)
    #     print("mkdir:" + results_dir)
    # else:
    #     print(colored('WARNING: already exists directory: ' + results_dir, 'green'))
    #
    #
    # '''********************************************************************'''
    # '''******************Generate PDF with all results*********************'''
    # '''********************************************************************'''
    #
    # if config.generate_pdf_flag:
    #     print("Generating pdf Plots in folder: ", results_dir)
    #     utils.generate_pdf(results_dir)

    experiments = dict()
    for dirName, subdirList, fileList in os.walk(dirPath):

        num_robots = "-1"
        timeout = "-1"
        elements = dirName.split("_")
        for e in elements:
            if e.startswith("robots"):
                num_robots = e.split("#")[-1]

            if e.startswith("timeout"):
                timeout = int(e.split("#")[-1])

        if num_robots == "-1" or timeout == -1:
            continue

        print(dirName)
        completed_areas = np.array([])
        for filename in os.listdir(dirName):
            if filename.endswith("taskLOG_client.tsv"):
                # print(colored(filename, 'green'))
                if not os.path.getsize(os.path.join(dirName, filename)) > 0:
                    print(colored("WARNING, empty file at:" + os.path.join(dirName, filename), 'red'), )
                    continue
                df = pd.read_csv(dirName + "/" + filename, sep="\t", header=None)
                completed_areas = np.append(completed_areas, df.shape[0])
                # print("Completed areas:", df.shape[0])
                # print(df)

        print("completed_areas:", completed_areas)
        # print(completed_areas.mean())
        # experiments[timeout] = completed_areas.mean()
        # break

        exp_dict = {config.folder_experiments+"#"+str(timeout): completed_areas}
        # wrt = csv.writer(open("output.csv", "a"))
        # for key, val in exp_dict.items():
        #     wrt.writerow([key, val])
        f = open("output.txt", "a")
        f.write(str(exp_dict)+'\n')
        f.close()

    # print(experiments)
    # experiments = OrderedDict(sorted(experiments.items()))
    # print(experiments)
    #
    # keys = experiments.keys()
    # values = experiments.values()
    #
    # plt.bar(keys, values)
    # plt.show()


if __name__ == '__main__':
    main()
