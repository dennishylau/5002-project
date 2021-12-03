import pandas as pd


def min_max_scale(series: pd.Series) -> pd.Series:
    'Returns: scaled series with range [0,1]'
    series_min = series.min()
    return (series - series_min) / (series.max() - series_min)
