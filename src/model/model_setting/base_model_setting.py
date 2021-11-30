from ast import Call
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Callable, Any
from model.time_series import TimeSeries
if TYPE_CHECKING:
    from model.anomaly import Anomaly


def cache(func: Callable):
    'Decorator for caching a calculation step to self.cache'

    def get_cache(self, *args, **kwargs):
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
    def add_df_column(self, ts: TimeSeries):
        '''
        Add extra columns to the ts obj's DataFrame for plotting
        '''
        raise NotImplementedError
