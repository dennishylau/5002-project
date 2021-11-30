from dataclasses import dataclass, field
from copy import deepcopy
from typing import Optional, TYPE_CHECKING
from functools import cached_property
import pandas as pd
from util.file_parser import parse_anomaly_start, parse_txt
from util.period_finder import find_period, int_plot_peaks_valleys
from util.plot import int_plot, int_plot_color_region
from plotly.graph_objects import Figure
if TYPE_CHECKING:
    from .model_setting import BaseModelSetting


@dataclass
class TimeSeries:
    '''
    Time series obj that init with base_path and filename.
    base_path: path to file with ending slash.
    filename: filename of data file.
    prediction_models: objs that conform to `BaseModelSetting`
    period_d_min: min value of search space for period
    period_d_max: max value of search space for period
    '''

    base_path: str
    filename: str
    prediction_models: list['BaseModelSetting'] = field(default_factory=list)
    period_d_min: int = 100
    period_d_max: int = 300

    def __post_init__(self):
        '''
        Ensure `prediction_models` and all objs within are deep copy of
        the original instance, so BaseModelSetting objects can be stateful
        for better performance
        '''
        self.prediction_models = deepcopy(self.prediction_models)

    # cache attr
    fig: Optional[Figure] = field(
        default=None, init=False, repr=False)

    # total ordering
    def __eq__(self, other):
        return self.filename == other.filename

    def __lt__(self, other):
        return self.filename < other.filename

    @cached_property
    def anomaly_start(self) -> int:
        'Parse filename and return starting point of anomaly'
        return parse_anomaly_start(self.filename)

    @cached_property
    def df(self) -> pd.DataFrame:
        '''
        Get DataFrame from base path and filename.
        Column name hardcoded as `series`.
        '''
        return parse_txt(
            file_path=self.base_path + self.filename,
            column_name='series')

    @cached_property
    def anomaly_series(self) -> pd.Series:
        'The pd.series after the `anomaly_start` point'
        return self.df.series[self.anomaly_start:]

    @cached_property
    def period(self) -> int:
        'Get period of the signal'
        return find_period(
            self.df.series, self.period_d_min, self.period_d_max)

    @cached_property
    def int_plot_color_region_width(self) -> int:
        '''
        How wide the colored region will be on the interactive plot in
        absolute terms (index of the DataFrame)
        '''
        return int(self.df.shape[0] * 0.01)

    def int_plot_peaks_valleys(self, **kwargs) -> Figure:
        'Interactive plot of period finder'
        fig = int_plot_peaks_valleys(
            title=self.filename,
            df=self.df,
            d_min=self.period_d_min,
            d_max=self.period_d_max, **kwargs)
        return fig

    def int_plot(
            self,
            force_recreate: bool = False) -> Figure:
        '''
        produce interactive plot of the different methods, and cache to the object instance.
        '''
        # check if there is a cached fig
        if self.fig is not None and not force_recreate:
            return self.fig

        # additional series to be plotted
        for prediction_model in self.prediction_models:
            # mutate and add columns to self.df
            prediction_model.add_to_df(ts=self)

        # base plot
        fig = int_plot(self.filename, self.df)

        for prediction_model in self.prediction_models:
            for anomaly in prediction_model.anomalies(ts=self):
                int_plot_color_region(
                    fig,
                    anomaly=anomaly,
                    width=self.int_plot_color_region_width,
                    annotation=prediction_model.annotation,
                    color=prediction_model.color)

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
