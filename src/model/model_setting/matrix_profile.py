import matrixprofile as mp
from typing import Any, TYPE_CHECKING
from model.anomaly import Anomaly
from .base_model_setting import BaseModelSetting
if TYPE_CHECKING:
    from model.time_series import TimeSeries


class MatrixProfile(BaseModelSetting):
    '''
    ts: the `TimeSeries` obj, with column called `series`
    annotation: the name of the colored region on plot
    color: color of the region on plot
    num_periods: window_size = TimeSeries.period * num_periods
    '''
    ts: 'TimeSeries'
    annotation: str
    color: str
    num_periods: int

    @property
    def window_size(self):
        return self.ts.period * self.num_periods

    def cal_profile(self, type: str = 'discords') -> dict[str, Any]:
        '''
        type: either `motifs` or `discords`
        returns: dict of motif or discords
        '''
        anomaly_series = self.ts.anomaly_series
        # calculating the matrix profile with window size'4'
        profile = mp.compute(anomaly_series.to_numpy(), self.window_size)
        if type == 'motifs':
            profile = mp.discover.motifs(profile, k=self.window_size)
        else:
            profile = mp.discover.discords(profile)
        return profile

    @property
    def anomalies(self) -> list['Anomaly']:
        '''
        Concrete implementation of BaseModelSetting.anomalies.
        Returns: list of `Anomaly` obj. Confidence is not calculated and has value `None`.
        TODO: cal confidence
        '''
        profile_dict: dict[str, Any] = self.cal_profile(type='discords')
        # get indexes relative to the anomaly start point
        relative_idxs: list[int] = profile_dict['discords']
        # get absolute indexes
        discords: list[int] = [
            self.ts.anomaly_start + i for i in relative_idxs]
        return [Anomaly(idx, None) for idx in discords]
