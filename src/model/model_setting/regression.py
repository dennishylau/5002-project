# %%
import numpy as np
import pandas as pd
import torch
from torch import nn
from dataclasses import dataclass
from .base_model_setting import BaseModelSetting, cache
from model.time_series import TimeSeries
from model.anomaly import Anomaly


@dataclass
class Regression(BaseModelSetting):
    PT_PATH = '../regression_params/'

    def anomalies(self, ts: TimeSeries) -> list['Anomaly']:
        try:
            idx, conf = self.confidence(ts)
            return [Anomaly(idx, conf)]
        except ValueError:
            # more than one anormaly found
            return []

    @cache
    def residual(self, ts: TimeSeries, padding=True) -> pd.Series:

        series = ts.df.series.to_numpy()
        x_left, x_right, y = self.create_regression_set(series)

        model = ConvModel()
        compute_loss = nn.MSELoss()
        params = torch.load(self.PT_PATH + ts.filename + '.pt')
        model.load_state_dict(params)

        residual = []
        for i in range(len(y)):
            batch_y = torch.tensor(y[i], dtype=torch.float32).view(1, 1)
            batch_left = x_left[i]
            batch_left = torch.tensor(
                batch_left, dtype=torch.float32).view(
                1, 1, -1)
            batch_right = x_right[i]
            batch_right = torch.tensor(
                batch_right, dtype=torch.float32).view(
                1, 1, -1)
            pred = model(batch_left, batch_right)
            loss = compute_loss(pred, batch_y)
            residual.append(float(loss))

        residual_mean = np.average(residual)
        if padding:
            pad = [residual_mean for _ in range(64 + 16)]
            result = pad + residual + pad
        else:
            result = residual

        return pd.Series(result)

    def create_regression_set(self, arr, outer=64, inner=16, delta=1):
        x_right = []
        x_left = []
        y = []
        pivot = outer + inner
        while pivot < len(arr) - outer - inner:
            left = arr[pivot - inner - outer:pivot - inner]
            right = arr[pivot + inner:pivot + inner + outer]
            x_left.append(left)
            x_right.append(right)
            y.append(arr[pivot])
            pivot += delta
        return x_left, x_right, y


class ConvModel(nn.Module):

    def __init__(self, linear_size=1440):
        super(ConvModel, self).__init__()
        self.conv_left = nn.Conv1d(1, 12, 5)
        self.conv_right = nn.Conv1d(1, 12, 5)
        self.linear = nn.Linear(linear_size, 1)
        self.act = nn.LeakyReLU(negative_slope=0.05)

    def forward(self, left, right):
        left = self.act(self.conv_left(left))
        right = self.act(self.conv_right(right))
        n, c, f = left.size()
        left = left.view(n, c * f)
        right = right.view(n, c * f)
        return self.linear(torch.cat([left, right], dim=1))
