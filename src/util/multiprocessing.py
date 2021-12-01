from multiprocessing import Pool
from tqdm import tqdm
from model.time_series import TimeSeries
from functools import partial
from typing import Iterable, Callable, TypeVar
from model.model_setting import BaseModelSetting


# Declare Generic
T = TypeVar('T')


def precal(
        filename: str,
        base_path: str,
        prediction_models: list[BaseModelSetting]) -> TimeSeries:
    '''
    Call TimeSeries.int_plot() to trigger calculation for all `BaseModelSetting`
    '''
    ts = TimeSeries(base_path, filename)
    ts.prediction_models = prediction_models
    ts.int_plot()
    return ts


def int_plot_peaks_valleys(
        filename: str,
        base_path: str):
    ts = TimeSeries(base_path, filename)
    ts.int_plot_peaks_valleys(
        export=True, export_path='../output/period_finder/')


def int_plot_inference(
        filename: str,
        base_path: str,
        prediction_models: list[BaseModelSetting]) -> TimeSeries:
    ts = precal(filename, base_path, prediction_models)
    ts.int_plot_export_html(f'../output/inference/{ts.filename[:-4]}.html')
    return ts


def mp_process(
        *,
        func: Callable[..., T],
        iterable: Iterable,
        chunksize: int = 1,
        **kwargs) -> list[T]:
    '''
    Wrapper for multiprocessing a func, returns an unordered list of generic type T.
    func: takes a single positional arg (an element from `iterable`) and `**kwargs`, returns T. Element from `iterable` must be the first arg.
    iterable: each element will be fed into `func` as positional arg
    chunksize: chunksize of imap_unordered, modify according to task
    '''
    T_list: list[T] = []
    # by default uses num of cpu
    with Pool() as pool:
        # dynamic progress bar for ipython
        for ts in tqdm(
            pool.imap_unordered(
                partial(func, **kwargs),
                iterable,
                chunksize=chunksize
            ),
            total=len(iterable)
        ):
            T_list.append(ts)
    return T_list
