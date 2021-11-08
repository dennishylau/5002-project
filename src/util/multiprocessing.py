from multiprocessing import Pool
from tqdm import tqdm
from model.time_series import TimeSeries
from functools import partial


def precal(base_path: str, filename: str) -> TimeSeries:
    ts = TimeSeries(base_path, filename)
    # ts.anomalies_precal()
    ts.int_plot()
    return ts


def mp_precal(base_path, filenames) -> list[TimeSeries]:
    'Multiprocess TimeSeries precal with progress bar'
    ts_list: list[TimeSeries] = []
    # by default uses num of cpu
    with Pool() as pool:
        # dynamic progress bar for ipython
        for ts in tqdm(
            pool.imap_unordered(partial(precal, base_path), filenames),
            total=len(filenames)
        ):
            ts_list.append(ts)
    return sorted(ts_list)
