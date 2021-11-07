# %%
# %matplotlib widget
# %reload_ext autoreload
# %autoreload 2

# %%
import os
from tqdm import tqdm
from model.time_series import TimeSeries

# %%

# base setup

BASE_PATH = '../data-sets/KDD-Cup/data/'
EXPORT_PATH = '../output/plot/2nd_diff/'

filenames: list[str] = sorted(os.listdir(BASE_PATH))
filenames = [i for i in filenames if 'txt' in i]
ts: list[TimeSeries] = [TimeSeries(BASE_PATH, i) for i in filenames]
# %%
for t in tqdm(ts[0:10]):
    t.int_plot_show()
