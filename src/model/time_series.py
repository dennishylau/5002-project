from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd
from model.anomaly import Anomaly
from util.file_parser import parse_anomaly_start, parse_txt
from util.period_finder import find_period, int_plot_peaks_valleys
from util.differencing import transform_2nd_order, confidence_2nd_diff
from util.matrix_profile import cal_matrix_profile
from util.plot import int_plot, int_plot_color_region
from plotly.graph_objects import Figure


@dataclass
class TimeSeries:
    base_path: str
    filename: str
    period_d_min: int = 100
    period_d_max: int = 300
    num_periods: int = 10
    fig: Optional[Figure] = None

    @property
    def anomaly_start(self) -> int:
        'Parse filename and return starting point of anomaly'
        return parse_anomaly_start(self.filename)

    @property
    def df(self) -> pd.DataFrame:
        '''
        Get DataFrame from base path and filename.
        Column name hardcoded as `series`.
        '''
        return parse_txt(self.base_path + self.filename)

    @property
    def period(self) -> int:
        'Get period of the signal'
        return find_period(
            self.df.series, self.period_d_min, self.period_d_max)

    @property
    def int_plot_color_region_width(self) -> int:
        '''
        How wide the colored region will be on the interactive plot in absolute terms (index of the DataFrame)
        '''
        return int(self.df.shape[0] * 0.01)

    @property
    def anomalies_2nd_diff(self) -> list[Anomaly]:
        'Returns: list of `Anomaly` obj. A list is returned for interoperability, even though the underlying `confidence_2nd_diff()` will return an empty list unless there is a unique result.'
        s_2nd_order = transform_2nd_order(self.df)
        try:
            idx, conf = confidence_2nd_diff(self.filename, s_2nd_order)
            return [Anomaly(idx, conf)]
        except ValueError:
            # more than one anormaly found
            return []

    @property
    def anomalies_matrix_profile(self) -> list[Anomaly]:
        '''
        Returns: list of `Anomaly` obj. Confidence is not calculated and has value `None`.
        TODO: cal confidence
        '''
        period = self.period
        profile_dict: dict[str, Any] = cal_matrix_profile(
            filename=self.filename,
            series=self.df.series,
            window_size=period * self.num_periods
        )
        # get indexes relative to the anomaly start point
        relative_idxs: list[int] = profile_dict['discords']
        # get absolute indexes
        discords: list[int] = [self.anomaly_start + i for i in relative_idxs]
        return [Anomaly(idx, None) for idx in discords]

    def int_plot_peaks_valleys(self):
        'Interactive plot of period finder'
        fig = int_plot_peaks_valleys(
            title=self.filename + ' Peaks Valleys',
            df=self.df,
            d_min=self.period_d_min,
            d_max=self.period_d_max)
        fig.show()

    def df_add_2nd_diff(self, df) -> pd.DataFrame:
        'Return a new df with 2nd order diff values'
        df = self.df.copy()
        s_2nd_order = transform_2nd_order(self.df)
        df['2nd_diff'] = s_2nd_order
        return df

    def int_plot(
            self,
            plot_2nd_diff: bool = True,
            force_recreate: bool = False) -> Figure:
        '''
        produce interactive plot of the different methods, and cache to the object instance.
        '''
        # check if there is a cached fig
        if self.fig is not None and not force_recreate:
            return self.fig

        # additional series to be plotted
        df = self.df
        if plot_2nd_diff:
            df = self.df_add_2nd_diff(df)
        # base plot
        fig = int_plot(self.filename, df)

        # plot 2nd order diff
        for anomaly in self.anomalies_2nd_diff:
            int_plot_color_region(
                fig,
                anomaly=anomaly,
                width=self.int_plot_color_region_width,
                annotation='2nd Diff',
                color='red')
        # plot matrix profile
        for anomaly in self.anomalies_matrix_profile:
            int_plot_color_region(
                fig,
                anomaly=anomaly,
                width=self.int_plot_color_region_width,
                annotation='mp',
                color='blue')

        # cache fig to instance
        self.fig = fig
        return fig

    def int_plot_export_html(self, file_path: str):
        '''
        Generate `int_plot()` and export html to file_path.
        file_path: should include `.html` as file extension.
        '''
        if not file_path.endswith('.html'):
            raise ValueError('file_path must have html extension.')
        fig = self.int_plot()
        fig.write_html(file_path)

    def int_plot_show(self):
        'Generate `int_plot()` and show in ipython.'
        fig = self.int_plot()
        fig.show()
