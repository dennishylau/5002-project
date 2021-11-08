from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model.time_series import TimeSeries
    from model.anomaly import Anomaly


@dataclass
class BaseModelSetting(ABC):
    '''
    Abstract class of ModelSetting.
    ts: the `TimeSeries` obj, with column called `series`
    annotation: the name of the colored region on plot
    color: color of the region on plot
    '''
    ts: 'TimeSeries'
    annotation: str
    color: str

    @property
    def anomalies(self) -> list['Anomaly']:
        pass
