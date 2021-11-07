import pandas as pd
import re


def parse_txt(
        file_path: str,
        column_name: str = 'series') -> pd.DataFrame:
    'Parse the input file'
    df = pd.read_csv(file_path, names=[column_name])
    try:
        df[column_name] = df[column_name].astype(float)
    except ValueError:
        df = pd.DataFrame(
            [i for i in df.loc[0, column_name].split(' ') if i != ''],
            columns=[column_name])
        df[column_name] = df[column_name].astype(float)

    return df


def parse_anomaly_start(filename: str) -> int:
    '''
    Parse the filename and return the earliest point in time 
    that may contain an anomaly
    '''

    regex = re.compile(r'^\d{3}_UCR_Anomaly_(?P<pos>\d+)\.txt$')
    idx: int = int(regex.search(filename).group(1))
    return idx
