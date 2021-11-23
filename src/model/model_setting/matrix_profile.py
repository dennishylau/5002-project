from dataclasses import dataclass
import matrixprofile as mp
from typing import Any, TYPE_CHECKING
import pandas as pd
from model.anomaly import Anomaly
from .base_model_setting import BaseModelSetting
# if TYPE_CHECKING:
from model.time_series import TimeSeries


@dataclass
class MatrixProfile(BaseModelSetting):
    '''
    annotation: the name of the colored region on plot
    color: color of the region on plot
    num_periods: window_size = TimeSeries.period * num_periods
    '''
    annotation: str
    color: str
    num_periods: int

    def anomalies(self, ts: TimeSeries) -> list['Anomaly']:
        '''
        Concrete implementation of BaseModelSetting.anomalies.
        Returns: list of `Anomaly` obj. Confidence is not calculated and has value `None`.
        TODO: cal confidence
        '''
        profile_dict: dict[str, Any] = self.cal_profile(ts, type='discords')
        # get indexes relative to the anomaly start point
        relative_idxs: list[int] = profile_dict['discords']
        # get absolute indexes
        discords: list[int] = [
            ts.anomaly_start + i for i in relative_idxs]
        return [Anomaly(idx, None) for idx in discords]

    def add_df_column(self, ts: TimeSeries):
        '''
        Add extra columns to the ts obj's DataFrame for plotting
        '''
        pass

    def window_size(self, ts: TimeSeries) -> int:
        return ts.period * self.num_periods

    def cal_profile(self, ts: TimeSeries, type: str = 'discords') -> dict[str, Any]:
        '''
        type: either `motifs` or `discords`
        returns: dict of motif or discords
        '''
        anomaly_series = ts.anomaly_series
        # cal window size
        window_size = self.window_size(ts)
        # calculating the matrix profile with window size'4'
        profile = mp.compute(anomaly_series.to_numpy(), window_size)
        if type == 'motifs':
            profile = mp.discover.motifs(profile, k=window_size)
        else:
            profile = mp.discover.discords(profile)
        return profile