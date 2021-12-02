from dataclasses import dataclass
from typing import Any
import matrixprofile as mp
import numpy as np
import pandas as pd
from model.anomaly import Anomaly
from model.time_series import TimeSeries
from .base_model_setting import BaseModelSetting, cache


@dataclass
class MatrixProfile(BaseModelSetting):
    '''
    annotation: the name of the colored region on plot
    color: color of the region on plot
    num_periods: window_size = TimeSeries.period * num_periods
    '''
    num_periods: int

    def anomalies(self, ts: TimeSeries) -> list['Anomaly']:
        '''
        Concrete implementation of BaseModelSetting.anomalies.
        Returns: list of `Anomaly` obj. Confidence is not calculated and has value `None`.
        TODO: cal confidence
        '''
        profile_dict: dict[str, Any] = self.cal_profile(ts)
        # get indexes relative to the anomaly start point
        relative_idxs: list[int] = profile_dict['discords']
        # get absolute indexes
        discords: list[int] = [
            ts.anomaly_start + i for i in relative_idxs]
        return [Anomaly(idx, None) for idx in discords]

    def residual(self, ts: TimeSeries) -> pd.Series:
        '''
        Add extra columns to the ts obj's DataFrame for plotting
        Returns: residual pandas series
        '''
        profile_dict: dict[str, Any] = self.cal_profile(ts)
        residual_mean = profile_dict['mp'].mean()
        lead_padding = np.full((ts.anomaly_start), residual_mean)
        trail_padding = np.zeros(profile_dict['w'] - 1) + residual_mean
        mp_adjusted = np.r_[lead_padding, profile_dict['mp'], trail_padding]
        return pd.Series(mp_adjusted)

    def window_size(self, ts: TimeSeries) -> int:
        'Cal window size: period * `num_periods`'
        return ts.period * self.num_periods

    @cache
    def cal_profile(self, ts: TimeSeries, type_: str = 'discords') -> dict[str, Any]:
        '''
        type: either `motifs` or `discords`
        returns: dict of motif or discords
        '''
        anomaly_series = ts.anomaly_series
        # cal window size
        window_size = self.window_size(ts)
        # calculating the matrix profile with window size'4'
        profile = mp.compute(anomaly_series.to_numpy(), window_size)
        if type_ == 'motifs':
            profile = mp.discover.motifs(profile, k=window_size)
        else:
            profile = mp.discover.discords(profile)
        return profile
