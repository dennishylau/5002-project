import os
import torch
from torch.optim import Adam
from torch import nn
import pandas as pd
from model.model_setting.regression import ConvModel


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


BASE_PATH = '../data-sets/KDD-Cup/data/'
fnames: list[str] = sorted(os.listdir(BASE_PATH))
fnames = [i for i in fnames if 'txt' in i]
for fname in fnames:
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
            batch_y = torch.tensor(
                y[i: i + batch_size],
                dtype=torch.float32).view(
                batch_size, 1).cuda()
            batch_left = x_left[i:i + batch_size]
            batch_left = torch.tensor(
                batch_left, dtype=torch.float32).view(
                batch_size, 1, -1).cuda()
            batch_right = x_right[i:i + batch_size]
            batch_right = torch.tensor(
                batch_right, dtype=torch.float32).view(
                batch_size, 1, -1).cuda()
            pred = model(batch_left, batch_right)
            loss = compute_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss)
        if j % 2 == 1:
            print(f'{j + 1} Epochs completed')
            print(f'Epoch Loss: {loss_sum * batch_size / len(y)}')

    model = model.cpu()
    torch.save(model.state_dict(), f'../regression_params/{fname}.pt')
