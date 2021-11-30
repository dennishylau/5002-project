from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pandas as pd
from typing import TYPE_CHECKING, Optional, Callable, Any, TypeVar
from model.time_series import TimeSeries
from util.scale import min_max_scale
if TYPE_CHECKING:
    from model.anomaly import Anomaly


# Declare Generic
T = TypeVar('T')


def cache(func: Callable[..., T]):
    'Decorator for caching a calculation step to self.cache'

    def get_cache(self, *args, **kwargs) -> T:
        if self.cache is None:
            # print('Uncached cal')
            self.cache = func(self, *args, **kwargs)
        return self.cache

    return get_cache


@dataclass
class BaseModelSetting(ABC):
    '''
    Abstract class of ModelSetting.
    annotation: the name of the colored region on plot
    color: color of the region on plot
    '''
    annotation: str
    color: str
    cache: Optional[Any] = field(default=None, init=False)

    @abstractmethod
    def anomalies(self, ts: TimeSeries) -> list['Anomaly']:
        'Return a list of Anomalies based on settings of this `BaseModelSetting` instance'
        raise NotImplementedError

    @abstractmethod
    def residual(self, ts: TimeSeries) -> pd.Series:
        '''
        Add extra columns to the ts obj's DataFrame for plotting.
        Returns: residual pandas series
        '''
        raise NotImplementedError

    def confidence(
            self,
            ts: TimeSeries) -> tuple[int, float]:
        '''
        The weight of the final ensemble, aka the degree of confidence,
        equals (highest peak of residual) / (2nd higest peak of residual)
        returns: tuple(index, confidence)
        '''
        anomaly_start = ts.anomaly_start
        # get the 2 largest absolute values
        series = self.residual(ts)
        unique_series = pd.Series(series[anomaly_start:].unique())
        values = unique_series.nlargest(2).to_list()
        largest = values[0]
        second = values[1]
        # cal confidence
        conf = largest / second
        # ensure largest is unique
        largest_series = unique_series[unique_series == largest]
        unique = largest_series.value_counts()[largest] == 1
        if not unique:
            raise ValueError('Largest value is not unique')
        # get the index of the value
        idxmax = anomaly_start + largest_series.index[0]
        return idxmax, conf

    def add_to_df(self, ts: TimeSeries):
        'Add residual series to ts.df'
        ts.df[self.annotation] = min_max_scale(self.residual(ts))

    def get_residual_peaks(self, ts: TimeSeries) -> dict[int, float]:
        '''
        Get peaks of residual series in the anomaly zone.
        Returns: dict[index, peak value]
        '''
        residual = self.residual(ts)[ts.anomaly_start:]
        pivot = 0
        peaks: dict[int, float] = {}
        while True:
            if pivot + ts.period > residual.size:
                peak_idx = residual[pivot:].idxmax()
                peaks[peak_idx] = residual.loc[peak_idx]
                break
            else:
                peak_idx = residual[pivot:pivot + ts.period].idxmax()
                peaks[peak_idx] = residual.loc[peak_idx]
                pivot += ts.period
        return peaks
