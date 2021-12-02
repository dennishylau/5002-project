# %%
# %matplotlib widget
# %reload_ext autoreload
# %autoreload 2

# %%
import os
from util.multiprocessing import precal, mp_process, int_plot_peaks_valleys, int_plot_inference
from model.model_setting import MatrixProfile, SecondOrderDiff, Regression
import pandas as pd
import pickle

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
sec_od = SecondOrderDiff(
    annotation='2nd Diff', color='red')
reg = Regression(
    annotation='cnn', color='green')
prediction_models = [mp1, mp10, sec_od, reg]


# %% without multiprocessing

# ts = TimeSeries(BASE_PATH, filenames[0])
# ts.prediction_models = prediction_models
# ts.int_plot_show()

# %% with multiprocessing

if __name__ == '__main__':
    # generate plots to eval period finder
    # mp_process(
    #     func=int_plot_peaks_valleys,
    #     iterable=filenames,
    #     base_path=BASE_PATH)

    # use multiprocessing to precal fields and pre-plot charts
    ts_list = mp_process(
        # func=precal,  # use precal for inference only
        func=int_plot_inference,  # use int_plot_inference for plots
        iterable=filenames,
        base_path=BASE_PATH,
        prediction_models=prediction_models)
    ts_list.sort()
# %% plot the first ts
    ts_list[0].int_plot_show()
# %% write submission output file
    ensemble = list(map(lambda x: x.ensemble(), ts_list))
    location = pd.Series([i.idx for i in ensemble])
    confidence = pd.Series([i.confidence for i in ensemble])
    df = pd.DataFrame({
        'Location of Anomaly': location,
        'Confidence': confidence})
    df.index += 1
    print(df)
    df['Location of Anomaly'].to_csv('../output/output.csv', index_label='No.')
# %% save ts objects by pickling for backup
    for ts in ts_list:
        with open(f'../output/ts_pickle/{ts.filename}.pk', 'wb') as file:
            pickle.dump(ts, file)
# %%
