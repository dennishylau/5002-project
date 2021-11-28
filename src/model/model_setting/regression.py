import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

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


def create_regression_set(arr, outer=64, inner=16, delta=1):
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


def get_regression_residuals(dataset, padding=True):

    BASE_PATH = './data-sets/KDD-Cup/data/'
    df = pd.read_csv(BASE_PATH + dataset, names=['values'])

    series = df['values'].to_numpy()
    x_left, x_right, y = create_regression_set(series)

    model = ConvModel()
    compute_loss = nn.MSELoss()
    params = torch.load('./regression_params/' + dataset + '.pt')
    model.load_state_dict(params)

    residual = []
    for i in range(len(y)):
        batch_y = torch.tensor(y[i], dtype=torch.float32).view(1, 1)
        batch_left = x_left[i]
        batch_left = torch.tensor(batch_left, dtype=torch.float32).view(1, 1, -1)
        batch_right = x_right[i]
        batch_right = torch.tensor(batch_right, dtype=torch.float32).view(1, 1, -1)
        pred = model(batch_left, batch_right)
        loss = compute_loss(pred, batch_y)
        residual.append(float(loss))

    if padding:
        pad = [0 for _ in range(64 + 16)]
        return pad + residual + pad
    else: 
        return residual


# Example
# residual = get_regression_residuals('007_UCR_Anomaly_4000.txt')