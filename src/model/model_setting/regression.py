from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch import nn, float32, tensor, cat
from model.time_series import TimeSeries
from .base_model_setting import BaseModelSetting, cache


@dataclass
class Regression(BaseModelSetting):
    'Regression Model Setting that uses CNN models to perform inference'

    PT_PATH = '../regression_params/'
    predictions: Optional[pd.Series] = field(default=None)

    @cache
    def residual(self, ts: TimeSeries) -> pd.Series:
        'Residual from CNN inference'
        series = ts.df.series.to_numpy()
        x_left, x_right, y = self.create_regression_set(series)

        model = ConvModel()
        params = torch.load(self.PT_PATH + ts.filename + '.pt')
        model.load_state_dict(params)

        predictions = []
        for i in range(len(y)):
            batch_left = tensor(x_left[i], dtype=float32).view(1, 1, -1)
            batch_right = tensor(x_right[i], dtype=float32).view(1, 1, -1)
            pred = model(batch_left, batch_right)
            predictions.append(pred.item())

        # align prediction
        pred_mean = np.average(predictions)
        pad = [pred_mean for _ in range(64 + 16)]
        predictions = pad + predictions + pad
        self.predictions = pd.Series(predictions)
        return (ts.series - self.predictions).abs()

    def create_regression_set(self, arr, outer=64, inner=16, delta=1):
        'Regression using left and right values'
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
    'CNN for training'

    def __init__(self, linear_size=1440):
        super(ConvModel, self).__init__()
        self.conv_left = nn.Conv1d(1, 12, 5)
        self.conv_right = nn.Conv1d(1, 12, 5)
        self.linear = nn.Linear(linear_size, 1)
        self.act = nn.LeakyReLU(negative_slope=0.05)

    def forward(self, left, right):
        'Forward prop'
        left = self.act(self.conv_left(left))
        right = self.act(self.conv_right(right))
        n, c, f = left.size()
        left = left.view(n, c * f)
        right = right.view(n, c * f)
        return self.linear(cat([left, right], dim=1))
