import pandas as pd
from scipy.fft import fft, ifft
import numpy as np
from model.time_series import TimeSeries
from .base_model_setting import BaseModelSetting, cache


class Fourier(BaseModelSetting):
    'Fourier transform, apply low-pass filter, then inverse fourier'
    @cache
    def residual(self, ts: TimeSeries) -> pd.Series:
        'Fourier residual'
        original = ts.df.series.to_numpy()
        threshold = 2 * ts.df.series.size / ts.period
        frequency = fft(original)
        low_pass_filtered = self.low_pass(frequency, threshold)
        low_pass_restore = ifft(low_pass_filtered)
        residual = np.abs(original - low_pass_restore)
        return pd.Series(residual)

    def low_pass(self, freq, threshold):
        'Low pass filter'
        through = freq[:threshold]
        pad = np.array([0 + 0j for _ in range(len(freq) - threshold)])
        return np.concatenate([through, pad])
