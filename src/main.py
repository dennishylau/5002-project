# %%
# %matplotlib widget
# %reload_ext autoreload
# %autoreload 2

# %%
import os
from model.time_series import TimeSeries
from util.multiprocessing import mp_precal

# %%

# base setup

BASE_PATH = '../data-sets/KDD-Cup/data/'

filenames: list[str] = sorted(os.listdir(BASE_PATH))
filenames = [i for i in filenames if 'txt' in i][:2]

if __name__ == '__main__':
    # use multiprocessing to precal fields and pre-plot charts
    ts_list: list[TimeSeries] = mp_precal(BASE_PATH, filenames)

    # do stuff here, or connect to interactive window in vscode
    # ts_list[0].int_plot_show()

# %%
