import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import numpy as np

def low_pass(freq, threshold):
    through = freq[:threshold]
    pad = np.array([0 + 0j for _ in range(len(freq)- threshold)])
    return np.concatenate([through, pad])

def high_pass(freq, threshold):
    through = freq[threshold:]
    pad = np.array([0 + 0j for _ in range(threshold)])
    return np.concatenate([pad, through])

def get_fourier_residuals(dataset, threshold):

    BASE_PATH = './data-sets/KDD-Cup/data/'
    df = pd.read_csv(BASE_PATH + dataset, names=['values'])
    original = df['values'].values

    plt.plot(original)
    plt.show()

    frequency = fft(original)
    low_pass_filtered = low_pass(frequency, threshold)
    low_pass_restore = ifft(low_pass_filtered)
    high_pass_filtered = high_pass(frequency, threshold)
    high_pass_restore = ifft(high_pass_filtered)

    return np.abs(original - low_pass_restore), np.abs(original - high_pass_restore)


# Example
# low_pass_residual, high_pass_residual = get_fourier_residuals('007_UCR_Anomaly_4000.txt', 200)
