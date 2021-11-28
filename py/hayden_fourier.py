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

BASE_PATH = './data-sets/KDD-Cup/data/'
df = pd.read_csv(BASE_PATH + '007_UCR_Anomaly_4000.txt', names=['values'])

plt.figure(figsize=[20, 5])
plt.plot(df['values'])
plt.show()

out = fft(df['values'].values)
filtered = high_pass(out, 200)
restore = ifft(filtered)

plt.figure(figsize=[20, 5])
plt.plot(np.abs(restore))
plt.show()