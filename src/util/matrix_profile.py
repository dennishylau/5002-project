import matrixprofile as mp
import pandas as pd
from .file_parser import parse_anomaly_start


def cal_matrix_profile(
        filename: str,
        series: pd.Series,
        window_size: int,
        type: str = 'discords') -> dict:
    '''
    window size: the period of the signal
    type: either `motifs` or `discords`
    returns: dict of motif or discords
    '''

    # parse the starting point of anomaly
    anomaly_start = parse_anomaly_start(filename)
    anomaly_seg = series[anomaly_start:]

    # calculating the matrix profile with window size'4'
    profile = mp.compute(anomaly_seg.to_numpy(), window_size)
    if type == 'motifs':
        profile = mp.discover.motifs(profile, k=window_size)
    else:
        profile = mp.discover.discords(profile)
    return profile
