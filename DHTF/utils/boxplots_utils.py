from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np


def print_dict(my_dict):
    for key, val in my_dict.items():
        print(colored(key, 'blue'), val)


def print_nested_dict(my_dict):
    for key, val in my_dict.items():
        print(colored(key, 'blue'))
        for k, v in val.items():
            print(k, v, end="\n\n")
            
                    
'''
# Boxplot from a single dict
'''
green_diamond = dict(markerfacecolor='g', marker='D')


def simple_boxplot(my_dict, fig_name):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=80)
    ax.boxplot(my_dict.values(), flierprops=green_diamond)
    ax.set_xticklabels(my_dict.keys())
    plt.ylim(-1, 44)
    plt.tight_layout()
    plt.savefig(fig_name+'.png')
    

def set_box_color(bp, color):
    """
    # Set boxplot rules
    """
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def grouped_3_boxplot(red, blue, mixed, fig_name):
    """
    # Boxplot from three dicts (red,blue,mixed)
    :param red: dict with red area values
    :param blue: dict with blue area values
    :param mixed: dict with red-blue area values
    :param fig_name: name of the output figure, saved in the current directory
    :return:
    """
    ticks = mixed.keys()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=80)

    bpl = plt.boxplot(red.values(), positions=np.array(range(len(red.values())))*3.0-0.4, sym='', widths=0.3)
    bpr = plt.boxplot(blue.values(), positions=np.array(range(len(blue.values())))*3.0+0.4, sym='', widths=0.3)
    bpc = plt.boxplot(mixed.values(), positions=np.array(range(len(mixed.values())))*3.0, sym='', widths=0.3)

    set_box_color(bpl, '#d95f02') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#7570b3')
    set_box_color(bpc, '#1b9e77')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#d95f02', label='Red')
    plt.plot([], c='#7570b3', label='Blue')
    plt.plot([], c='#1b9e77', label='Mixed')
    plt.legend(loc=2)

    # plt.xticks(range(0, 300, 5), ticks)
    # plt.xlim(-3, len(ticks)*3)
    plt.ylim(-1, 44)
    plt.xticks(range(0, len(ticks) * 3, 3), ticks)
    plt.xlim(-2, len(ticks)*3)

    ax.set_xticklabels(mixed.keys())
    ax.set_xlabel("Timeout[s]")
    ax.set_ylabel("Completed areas")

    plt.tight_layout()
    plt.savefig(fig_name + '.png')


def grouped_4_boxplot(rr, rb, br, bb, y_lim_k, fig_name):
    """
    # Boxplot from three dicts (red,blue,mixed)
    :param rr: dict with red-red area values
    :param rb: dict with red-blue area values
    :param br: dict with blue-red area values
    :param bb: dict with blue-blue area values
    :param fig_name: name of the output figure, saved in the current directory
    :param y_lim_k: the y_lim for the boxplot
    :return:
    """
    ticks = rr.keys()

    fig, ax = plt.subplots(figsize=(20, 5), dpi=80)

    bpRR = plt.boxplot(rr.values(), positions=np.array(range(len(rr.values()))) * 3.0 - 0.8, sym='', widths=0.35)
    bpRB = plt.boxplot(rb.values(), positions=np.array(range(len(rb.values()))) * 3.0 - 0.25, sym='', widths=0.35)
    bpBR = plt.boxplot(br.values(), positions=np.array(range(len(rr.values()))) * 3.0 + 0.25, sym='', widths=0.35)
    bpBB = plt.boxplot(bb.values(), positions=np.array(range(len(bb.values()))) * 3.0 + 0.8, sym='', widths=0.35)

    set_box_color(bpRR, '#d95f02')  # colors are from http://colorbrewer2.org/
    set_box_color(bpRB, '#1c9099')
    set_box_color(bpBR, '#2ca25f')
    set_box_color(bpBB, '#7570b3')

    # draw temporary rr and blue lines and use them to create a legend
    plt.plot([], c='#d95f02', label='RR')
    plt.plot([], c='#1c9099', label='RB')
    plt.plot([], c='#2ca25f', label='BR')
    plt.plot([], c='#7570b3', label='BB')
    plt.legend(loc=2)

    # plt.xticks(range(0, 300, 5), ticks)
    # plt.xlim(-3, len(ticks)*3)
    plt.ylim(-1, y_lim_k)
    plt.xticks(range(0, len(ticks) * 3, 3), ticks)
    plt.xlim(-2, len(ticks) * 3)

    ax.set_xticklabels(rr.keys())
    ax.set_xlabel("Timeout[s]")
    # ax.set_ylabel("Elapsed timeout for each kilobot")

    plt.tight_layout()
    print("Saving grouped_4_boxplot with name:", fig_name)
    plt.savefig(fig_name + '.png')
