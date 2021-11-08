from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import pandas as pd
from model.time_series import TimeSeries
if TYPE_CHECKING:
    from model.anomaly import Anomaly


@dataclass
class BaseModelSetting(ABC):
    '''
    Abstract class of ModelSetting.
    annotation: the name of the colored region on plot
    color: color of the region on plot
    '''
    annotation: str
    color: str

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
