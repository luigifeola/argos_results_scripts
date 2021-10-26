from utils import utils
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.special as sc
import pandas as pd
import os
import seaborn as sns


def weib_cdf(x, alpha, gamma):
    return (1 - np.exp(-np.power(x / alpha, gamma)))


def evaluate_convergence_time(times):
    conv_times = np.zeros(times.shape[0])
    #     print("Time shape", times.shape)
    for idx, elem in enumerate(times):
        if (elem[0] == 0):
            conv_times[idx] = elem[1]
        else:
            conv_times[idx] = elem.min()
    # c_time in ticks
    #     conv_time_batch = np.append(conv_time_batch, conv_times.max())
    return conv_times


# data = time vector
# censored = number o missing values
def KM_estimator(data, censored):
    '''K-M estimator'''
    n_est = np.asarray(range(0, data.size))[::-1] + censored  # array from 29 to 0
    RT_sync = []
    for i in range(n_est.size):
        if len(RT_sync) == 0:
            RT_sync.append((n_est[i] - 1) / n_est[i])
        else:
            RT_sync.append(RT_sync[-1] * ((n_est[i] - 1) / n_est[i]))
    #     print(RT_sync)
    F = 1 - np.asarray(RT_sync).reshape(-1, 1)
    #     print(F)
    return F


def weibull_plot(mean, std_dev, times_value, popt_weibull, F, figLabel, figPath, conv_time_estimation):
    fig, ax = plt.subplots(figsize=(20, 8), dpi=160, facecolor='w', edgecolor='k')
    '''Textbox with mu and sigma'''
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mean,),
        r'$\sigma=%.2f$' % (std_dev,)))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    y_weib = weib_cdf(times_value, popt_weibull[0], popt_weibull[1])
    error_weib = np.power(y_weib - np.squeeze(F), 2)
    plt.plot(times_value, y_weib, 'r', linewidth=5, label="Weibull Distribution")
    plt.plot(times_value, F, 'b', linewidth=5, label="K-M stats")
    plt.legend(loc=4)
    plt.ylim(0, 1)

    label = figLabel
    plt.title(label)
    plt.xlabel("Number of time steps")
    if conv_time_estimation:
        plt.ylabel("Synchronisation probability")
    else:
        plt.ylabel("Probability of passing over the target")
    #     plt.show()
    plt.savefig(figPath)
    plt.close(fig)


# TODO : this already exists in utils, maybe you can use just one of them
def plot_heatmap(dictionary, title, storage_dir, conv_time_estimation=-1):
    for key, value in sorted(dictionary.items()):
        # print("Key value: ", key, value)
        fig = plt.figure(figsize=(12, 8), dpi=80)
        dataFrame = pd.DataFrame.from_dict(value)
        reversed_df = dataFrame.iloc[::-1]
        if conv_time_estimation:
            ax = sns.heatmap(reversed_df, annot=True, fmt=".2e", vmin=1500, vmax=3000, cmap="viridis")
        else:
            ax = sns.heatmap(reversed_df, annot=True, fmt=".2e", vmin=10000, vmax=40000, cmap="viridis")
        ax.set_title(title + ", num_robots:%s" % key)
        ax.set_ylabel("alpha")
        ax.set_xlabel("rho")
        #         plt.show()
        # Salva su file
        file_name = title + "_%s_robots.png" % (key)
        plt.savefig(storage_dir + '/' + file_name)
        #     reversed_df.to_pickle(file_name[:-4] + ".pickle")
        plt.close(fig)


def evaluate_time_stats(folder, conv_time_dir, ftp_dir, conv_time_estimation, bound_is):
    mean_fpt_dict = dict()
    convergence_time_dict = dict()
    for dirName, subdirList, fileList in os.walk(folder):
        # print(dirName)
        num_robots = "0"
        rho = -1.0
        alpha = -1.0
        elements = dirName.split("_")
        for e in elements:
            if e.startswith("robots"):
                num_robots = e.split("#")[-1]
                if num_robots not in mean_fpt_dict:
                    mean_fpt_dict[num_robots] = dict()
                    convergence_time_dict[num_robots] = dict()

            if e.startswith("rho"):
                rho = float(e.split("#")[-1])
            if e.startswith("alpha"):
                alpha = float(e.split("#")[-1])

        if num_robots == "0" or rho == -1.0 or alpha == -1:
            continue

        #     print(num_robots)
        #     print(mean_fpt_dict)

        rho_str = str(rho)
        alpha_str = str(alpha)
        #     print("rho", rho_str)
        #     print("alpha", alpha_str)
        if rho_str not in mean_fpt_dict[num_robots]:
            mean_fpt_dict[num_robots][rho_str] = dict()
            #         print(mean_fpt_dict)

            convergence_time_dict[num_robots][rho_str] = dict()

        #         print(total_dict)
        # WARNING : di mettere alpha probabilmente non ce n'è bisogno
        #     if(alpha_str not in total_dict[num_robots][rho_str]):
        #         total_dict[num_robots][rho_str][alpha_str]=dict()
        #         mean_fpt_dict[num_robots][rho_str][alpha_str]=dict()
        #         convergence_time_dict[num_robots][rho_str][alpha_str]=dict()

        (num_experiment, df) = utils.load_pd_times(dirName, "times")

        df_times = df.values[:, 1:]
        convergence_times = evaluate_convergence_time(df_times)
        #     print(dirName)
        #     print(convergence_times.shape)

        #     print(df_times.shape)
        #     print("num experiments: ", num_experiment)

        if conv_time_estimation:
            '''Weibull distribution for Convergence Time'''

            # get the time in whitch each robot has at least info about the target
            convergence_time_batches = np.amax(convergence_times.reshape(num_experiment, -1), axis=1)
            #         print(dirName)

            # order convergence_time_batches in increasing order
            convergence_time_batches = convergence_time_batches[np.argsort(convergence_time_batches)]
            #         print(convergence_time_batches.shape)
            #         print(convergence_time_batches)
            figPath = conv_time_dir + '/' + "conv_time_robots_%s_alpha_%s_rho_%s.png" % (num_robots, alpha_str, rho_str)
            figLabel = "Convergence Time robots:%s alpha:%s, rho:%s.png" % (num_robots, alpha_str, rho_str)
            #         censored = 1
            censored = convergence_time_batches.size - np.count_nonzero(convergence_time_batches)
            if censored:
                times_value = convergence_time_batches[censored:].reshape(-1)
            else:
                censored = 1
                times_value = convergence_time_batches.reshape(-1)

            F = KM_estimator(times_value, censored)

            # popt_weibull[0] is alpha
            # popt_weibull[1] is gamma
            popt_weibull, _ = curve_fit(weib_cdf, xdata=times_value, ydata=np.squeeze(F), bounds=(0, [bound_is, 10]),
                                        method='trf')
            mean = sc.gamma(1 + (1. / popt_weibull[1])) * popt_weibull[0]
            #     print("mean",mean)
            std_dev = np.sqrt(popt_weibull[0] ** 2 * sc.gamma(1 + (2. / popt_weibull[1])) - mean ** 2)

            std_error = std_dev / np.sqrt(times_value.size)
            convergence_time_dict[num_robots][rho_str][alpha_str] = mean
            #     print(times_value.shape)
            weibull_plot(mean, std_dev, times_value, popt_weibull, F, figLabel, figPath, conv_time_estimation)

            convergence_time_dict = utils.sort_nested_dict(convergence_time_dict)
            plot_heatmap(convergence_time_dict, "Convergence Time", conv_time_dir, conv_time_estimation)

        else:
            ''' Weibull distribution for First Passage Time'''
            figPath = ftp_dir + '/' + "fpt_robots_%s_alpha_%s_rho_%s.png" % (num_robots, alpha_str, rho_str)
            figLabel = "fpt robots:%s alpha:%s, rho:%s.png" % (num_robots, alpha_str, rho_str)
            fpt = df.values[:, 1:2]
            censored = fpt.size - np.count_nonzero(fpt)
            fpt = fpt[np.argsort(fpt.reshape(-1))]
            times_value = fpt[censored:].reshape(-1)

            F = KM_estimator(times_value, censored)

            # popt_weibull[0] is alpha
            # popt_weibull[1] is gamma
            popt_weibull, _ = curve_fit(weib_cdf, xdata=times_value, ydata=np.squeeze(F), bounds=(0, [bound_is, 10]),
                                        method='trf')
            mean = sc.gamma(1 + (1. / popt_weibull[1])) * popt_weibull[0]
            mean_fpt_dict[num_robots][rho_str][alpha_str] = mean
            #     print("mean",mean)
            std_dev = np.sqrt(popt_weibull[0] ** 2 * sc.gamma(1 + (2. / popt_weibull[1])) - mean ** 2)

            std_error = std_dev / np.sqrt(times_value.size)

            #     print(times_value.shape)
            weibull_plot(mean, std_dev, times_value, popt_weibull, F, figLabel, figPath, conv_time_estimation)
            #             print(mean_fpt_dict, end="\n\n")

            mean_fpt_dict = utils.sort_nested_dict(mean_fpt_dict)
            plot_heatmap(mean_fpt_dict, "Average First Passage Time", ftp_dir, conv_time_estimation)

    #     print("Convergence Time")
    #     print(convergence_time_dict)
    #     print("Average First Passage Time")
    #     print(mean_fpt_dict)

    #     print("Convergence Time alpha")
    #     print("FPT alpha")
