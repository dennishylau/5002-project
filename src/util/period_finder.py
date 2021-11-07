import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.graph_objects import Figure as IntFigure
from .plot import int_plot, int_plot_marker


def find_peaks_with_gap(signal: list[float], period: int) -> np.ndarray:
    '''
    Given entire time series and a period candidate, find all peaks
    abs_pos: returns absolute position in time series if true, else relative position in segment
    '''
    idx_list: list[int] = []
    slices = int(len(signal) / period)
    for i in range(slices):
        start = i * period
        end = start + period
        segment = signal[start:end]
        maxima_pos: int = np.argmax(segment)
        idx_list.append(start + maxima_pos)
    return np.array(idx_list)


def find_period(signal: list[float], d_min: int, d_max: int) -> int:
    '''
    Assumes all time series has fixed period `d`
    Returns: period length
    '''
    score_dict: dict[int, float] = dict()
    # for distance d
    for d in range(d_min, d_max):
        # find peaks
        p = find_peaks_with_gap(signal, d)
        # find valleys
        v = find_peaks_with_gap(signal * -1, d)
        # calculate score
        pd = [p[i + 1] - p[i] for i in range(len(p) - 1)]
        vd = [v[i + 1] - v[i] for i in range(len(v) - 1)]
        s: float = np.min([np.std(pd), np.std(vd)]) / np.sqrt(d)
        score_dict[d] = s
    # return d with lowest variance
    opt_d: int = min(score_dict, key=score_dict.get)
    return opt_d


def plot_peaks(
        d_min: int,
        d_max: int,
        filename: str,
        df: pd.DataFrame,
        series_name: str = 'series') -> Figure:
    'Plot out all peaks based on period'
    # set plot backend
    pd.options.plotting.backend = 'matplotlib'
    # get peaks
    series: list[float] = df[series_name]
    period = find_period(series, d_min, d_max)
    peaks = find_peaks_with_gap(series, period)

    figure = plt.figure(figsize=(30, 6))
    plt.plot(series)
    plt.title(filename)
    plt.scatter(peaks, [df[series_name][i] for i in peaks], c='red')
    plt.show()
    return figure


def int_plot_peaks_valleys(
        title: str,
        df: pd.DataFrame,
        *,
        d_min: int,
        d_max: int,
        series_name: str = 'series',
        export: bool = False,
        export_path: str = './',
        show: bool = False) -> IntFigure:
    # get series
    series: list[float] = df[series_name]
    period = find_period(series, d_min, d_max)
    # find peaks and valleys
    peak_x = find_peaks_with_gap(
        df[series_name].to_numpy(),
        period)
    peak_y = df.iloc[peak_x][series_name].to_numpy()
    valley_x = find_peaks_with_gap(
        df.series.to_numpy() * -1, period)
    valley_y = df.iloc[valley_x][series_name].to_numpy()
    # base plot
    fig: IntFigure = int_plot(title, df)
    # add markers
    int_plot_marker(fig, peak_x, peak_y, 'red', 'peaks')
    int_plot_marker(fig, valley_x, valley_y, 'orange', 'valleys')
    if export is True:
        fig.write_html(f'{export_path}{title}_peaks_valleys.html')
    if show is True:
        fig.show()
    return fig
