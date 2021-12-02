import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import numpy as np

def low_pass(freq, threshold):
    through = freq[:threshold]
    pad = np.array([0 + 0j for _ in range(len(freq)- threshold)])
    return np.concatenate([through, pad])

def get_fourier_residuals(dataset, period=0, threshold=0):

    BASE_PATH = './data-sets/KDD-Cup/data/'
    df = pd.read_csv(BASE_PATH + dataset, names=['values'])
    original = df['values'].values

    if threshold == 0:
        threshold = 2 * len(original) / period

    frequency = fft(original)
    low_pass_filtered = low_pass(frequency, threshold)
    low_pass_restore = ifft(low_pass_filtered)

    return np.abs(original - low_pass_restore)

# low_pass_residual = get_fourier_residuals('012_UCR_Anomaly_15000.txt', threshold=100)
# plt.plot(low_pass_residual)
# plt.show()
