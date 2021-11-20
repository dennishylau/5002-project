# %%
# %matplotlib widget
# %reload_ext autoreload
# %autoreload 2

# %%
import os
from model.time_series import TimeSeries
from util.multiprocessing import precal, mp_process
from model.model_setting import MatrixProfile, SecondOrderDiff #, SingularSpectrumTransformation

# %%

# base setup

BASE_PATH = '../data-sets/KDD-Cup/data/'
filenames: list[str] = sorted(os.listdir(BASE_PATH))
filenames = [i for i in filenames if 'txt' in i]

# declare the prediction models to be used
mp1 = MatrixProfile(
    annotation='mp1', color='blue', num_periods=1)
mp10 = MatrixProfile(
    annotation='mp10', color='brown', num_periods=10)
sec_od1 = SecondOrderDiff(
    annotation='2nd Diff', color='red')
#banpei = SingularSpectrumTransformation()
#    annotation='SST', color='green')
prediction_models = [mp1, mp10, sec_od1] #, banpei]


# %% without multiprocessing

# ts = TimeSeries(BASE_PATH, filenames[0])
# ts.prediction_models = prediction_models
# ts.int_plot_show()

# %% with multiprocessing

if __name__ == '__main__':
    # use multiprocessing to precal fields and pre-plot charts
    ts_list = mp_process(
        func=precal,
        iterable=filenames,
        base_path=BASE_PATH,
        prediction_models=prediction_models)
    ts_list.sort()

    # do stuff here, or connect to interactive window in vscode
    # ts_list[0].int_plot_show()

# %%
