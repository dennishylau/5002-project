from ast import excepthandler
import pandas as pd
import matplotlib.pyplot as plt
from plotly.graph_objects import Figure as IntFigure
from .file_parser import parse_anomaly_start, parse_txt
from util.plot import int_plot


def transform_2nd_order(
        df: pd.DataFrame,
        series_name: str = 'series') -> pd.Series:
    '''
    Returns: a Series with called `2nd_diff` from transformation
    '''
    # take differencing
    df_t1 = df.copy()
    df_t1.index -= 1
    df_t2 = df_t1.copy()
    df_t2.index -= 1

    df_did = df.join(df_t1, rsuffix='_t1').join(df_t2, rsuffix='_t2')
    df_did['series_did1'] = df_did.series_t1 - df_did[series_name]
    series: pd.Series = df_did.series_t2 - 2 * df_did.series_t1 + df_did.series
    series.name = '2nd_diff'
    return series


def confidence_2nd_diff(filename: str, series: pd.Series) -> tuple[int, float]:
    '''
    The weight of the final ensemble, aka the degree of confidence, equals (highest peak of residual) / (2nd higest peak of residual)
    series: the result from differencing
    returns: tuple(index, confidence)
    '''

    anomaly_start = parse_anomaly_start(filename)
    anomaly_seg = series[anomaly_start:]
    # get the 2 largest absolute values
    abs_series = pd.Series(anomaly_seg.abs().unique())
    values = abs_series.nlargest(2).to_list()
    largest = values[0]
    second = values[1]
    # cal confidence
    conf = largest / second
    # ensure largest is unique
    largest_series = abs_series[abs_series == largest]
    unique = largest_series.value_counts()[largest] == 1
    if not unique:
        raise ValueError('Largest value is not unique')
    # get the index of the value
    idxmax = anomaly_start + largest_series.index[0]
    return idxmax, conf
