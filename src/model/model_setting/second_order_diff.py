from dataclasses import dataclass
import pandas as pd
from model.anomaly import Anomaly
from .base_model_setting import BaseModelSetting
from model.time_series import TimeSeries


@dataclass
class SecondOrderDiff(BaseModelSetting):
    '''
    annotation: the name of the colored region on plot
    color: color of the region on plot
    '''
    annotation: str
    color: str

    def anomalies(self, ts: TimeSeries) -> list['Anomaly']:
        '''
        Concrete implementation of BaseModelSetting.anomalies.
        A list is returned for interoperability, even though the underlying `confidence_2nd_diff()` will return an empty list unless there is a unique result.
        Returns: list of `Anomaly` obj.
        '''
        s_2nd_order = self.transform_2nd_order(ts)
        try:
            idx, conf = self.confidence_2nd_diff(ts, s_2nd_order)
            return [Anomaly(idx, conf)]
        except ValueError:
            # more than one anormaly found
            return []

    def add_df_column(self, ts: TimeSeries):
        '''
        Add extra columns to the ts obj's DataFrame for plotting
        '''
        ts.df['2nd_diff'] = self.transform_2nd_order(ts)

    def transform_2nd_order(self, ts: TimeSeries) -> pd.Series:
        '''
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
        series.name = '2nd_diff'
        return series.abs()

    def confidence_2nd_diff(
            self,
            ts: TimeSeries,
            diff_series: pd.Series) -> tuple[int, float]:
        '''
        The weight of the final ensemble, aka the degree of confidence,
        equals (highest peak of residual) / (2nd higest peak of residual)
        diff_series: the 2nd order diff series
        returns: tuple(index, confidence)
        '''
        anomaly_start = ts.anomaly_start
        # get the 2 largest absolute values
        unique_series = pd.Series(diff_series[anomaly_start:].unique())
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
