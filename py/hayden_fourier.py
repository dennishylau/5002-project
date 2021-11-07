import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft

BASE_PATH = './data-sets/KDD-Cup/data/'
df = pd.read_csv(BASE_PATH + '007_UCR_Anomaly_4000.txt', names=['values'])

plt.figure(figsize=[20, 5])
plt.plot(df['values'])
plt.show()

out = fft(df['values'].values)
out = out[:len(out) // 2]
abs_out = [abs(o) for o in out]

plt.figure(figsize=[20, 5])
plt.plot(abs_out)
plt.show()