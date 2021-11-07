# %matplotlib widget
# %reload_ext autoreload
# %autoreload 2

# %%
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from util.matrix_profile import cal_matrix_profile
from util.period_finder import find_period, int_plot_peaks_valleys
from util.differencing import transform_2nd_order, confidence_2nd_diff
from util.file_parser import parse_anomaly_start, parse_txt
from util.plot import int_plot, int_plot_color_region

# %%

# base setup

BASE_PATH = '../data-sets/KDD-Cup/data/'
EXPORT_PATH = '../output/plot/2nd_diff/'

filenames = sorted(os.listdir(BASE_PATH))
filenames = [i for i in filenames if 'txt' in i]

# %%

# %%

figs = []

for filename in tqdm(filenames[0:5]):
    # find anomaly start
    start = parse_anomaly_start(filename)

    df = parse_txt(BASE_PATH + filename)

    # 2nd order diff
    s_2nd_order = transform_2nd_order(df)
    try:
        idx, conf = confidence_2nd_diff(filename, s_2nd_order)
    except ValueError:
        idx, conf = None, None

    # matrix profile
    d_min = 100
    d_max = 300
    num_periods = 10
    period = find_period(df.series, d_min, d_max)
    discords: list[int] = cal_matrix_profile(
        filename, df.series,
        window_size=period * num_periods)['discords']
    discords = [start + i for i in discords]

    # Add additional series to be plotted
    df['2nd_diff'] = s_2nd_order

    # plot period finder
    # int_plot_peaks_valleys(filename + ' Period', df,
    #                        d_min=100, d_max=300).show()

    # base plot
    fig = int_plot(filename, df)
    # set color region width
    color_width = df.shape[0] * 0.01
    # plot 2nd order diff
    int_plot_color_region(fig, idx - color_width,
                          idx + color_width, '2nd Diff', 'red')
    # plot matrix profile
    for discord in discords:
        int_plot_color_region(fig, discord - color_width,
                              discord + color_width, 'mp', 'blue')

    figs.append(fig)

for fig in figs:
    fig.show()

# %%
