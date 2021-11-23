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
    # plt.figure(figsize=[20, 5])
    # plt.plot(df['values'])
    # plt.show()

    series = df['values'].to_numpy()
    x_left, x_right, y = create_regression_set(series)

    model = ConvModel()
    compute_loss = nn.MSELoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=0.0005)

    batch_size = 64
    num_epoch = 15

    for j in range(num_epoch):
        print(f'Training {j + 1} Epoch...')
        loss_sum = 0
        for i in range(0, len(y) - batch_size, batch_size):
            optimizer.zero_grad()
            batch_y = torch.tensor(y[i:i + batch_size], dtype=torch.float32).view(batch_size, 1)
            batch_left = x_left[i:i + batch_size]
            batch_left = torch.tensor(batch_left, dtype=torch.float32).view(batch_size, 1, -1)
            batch_right = x_right[i:i + batch_size]
            batch_right = torch.tensor(batch_right, dtype=torch.float32).view(batch_size, 1, -1)
            pred = model(batch_left, batch_right)
            loss = compute_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss)
        print(f'Epoch Loss: {loss_sum}')

    # residual = []
    # for i in range(len(y)):
    #     batch_y = torch.tensor(y[i], dtype=torch.float32).view(1, 1)
    #     batch_left = x_left[i]
    #     batch_left = torch.tensor(batch_left, dtype=torch.float32).view(1, 1, -1)
    #     batch_right = x_right[i]
    #     batch_right = torch.tensor(batch_right, dtype=torch.float32).view(1, 1, -1)
    #     pred = model(batch_left, batch_right)
    #     loss = compute_loss(pred, batch_y)
    #     residual.append(float(loss))

    torch.save(model.state_dict(), f'./regression_params/{fname}.pt')

    # residual = np.array(residual)
    # peak_index = residual.argmax()
    # peak = residual[peak_index]

    # plt.figure(figsize=[20, 5])
    # plt.plot(residual)
    # plt.scatter([peak_index], [peak], c='red')
    # plt.show()