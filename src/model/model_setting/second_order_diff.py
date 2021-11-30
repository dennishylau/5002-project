from dataclasses import dataclass
import pandas as pd
from model.anomaly import Anomaly
from .base_model_setting import BaseModelSetting, cache
from model.time_series import TimeSeries


@dataclass
class SecondOrderDiff(BaseModelSetting):
    '''
    annotation: the name of the colored region on plot
    color: color of the region on plot
    '''

    def anomalies(self, ts: TimeSeries) -> list['Anomaly']:
        '''
        Concrete implementation of BaseModelSetting.anomalies.
        A list is returned for interoperability, even though the underlying `confidence_2nd_diff()` will return an empty list unless there is a 
        unique result.
        Returns: list of `Anomaly` obj.
        '''
        try:
            idx, conf = self.confidence(ts)
            return [Anomaly(idx, conf)]
        except ValueError:
            # more than one anormaly found
            return []

    @cache
    def residual(self, ts: TimeSeries) -> pd.Series:
        '''
        Add extra columns to the ts obj's DataFrame for plotting
        Returns: a Series with called `2nd_diff` from transformation in absolute
        value
        '''
        # take differencing
        df = ts.df
        df_t1 = df.copy()
        df_t1.index -= 1
        df_t2 = df_t1.copy()
        df_t2.index -= 1

        df_did = df.join(df_t1, rsuffix='_t1').join(df_t2, rsuffix='_t2')
        df_did['series_did1'] = df_did.series_t1 - df_did.series
        series: pd.Series = df_did.series_t2 - 2 * df_did.series_t1 + df_did.series
        series = series.abs()
        series.name = '2nd_diff'
        return series
