import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch import nn
import os

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

BASE_PATH = './data-sets/KDD-Cup/data/'
for fname in os.listdir('./data-sets/KDD-Cup/data/'):
    df = pd.read_csv(BASE_PATH + fname, names=['values'])
    print('Now processing:', fname)

    series = df['values'].to_numpy()
    x_left, x_right, y = create_regression_set(series)

    model = ConvModel().cuda()
    compute_loss = nn.L1Loss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=0.001)

    batch_size = 96
    num_epoch = 20

    for j in range(num_epoch):
        loss_sum = 0
        for i in range(0, len(y) - batch_size, batch_size):
            optimizer.zero_grad()
            batch_y = torch.tensor(y[i:i + batch_size], dtype=torch.float32).view(batch_size, 1).cuda()
            batch_left = x_left[i:i + batch_size]
            batch_left = torch.tensor(batch_left, dtype=torch.float32).view(batch_size, 1, -1).cuda()
            batch_right = x_right[i:i + batch_size]
            batch_right = torch.tensor(batch_right, dtype=torch.float32).view(batch_size, 1, -1).cuda()
            pred = model(batch_left, batch_right)
            loss = compute_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss)
        if j % 2 == 1:
            print(f'{j + 1} Epochs completed')
            print(f'Epoch Loss: {loss_sum * batch_size / len(y)}')

    model = model.cpu()
    torch.save(model.state_dict(), f'./regression_params/{fname}.pt')

    # residual = []
    # preds = []
    # for i in range(len(y)):
    #     batch_y = torch.tensor(y[i], dtype=torch.float32).view(1, 1)
    #     batch_left = x_left[i]
    #     batch_left = torch.tensor(batch_left, dtype=torch.float32).view(1, 1, -1)
    #     batch_right = x_right[i]
    #     batch_right = torch.tensor(batch_right, dtype=torch.float32).view(1, 1, -1)
    #     pred = model(batch_left, batch_right)
    #     error = torch.abs(pred - batch_y).view(1)
    #     preds.append(float(pred.view(1)))
    #     residual.append(float(error))

    # plt.plot(y)
    # plt.show()

    # plt.plot(preds)
    # plt.show()

    # plt.plot(residual)
    # plt.show()

    # y_mean = sum(y) / len(y)
    # y_mad = sum([abs(yi - y_mean) for yi in y]) / len(y)
    # print('Mean model L1 Error:', sum(residual) / len(residual))
    # print('Data y MAD:', y_mad)