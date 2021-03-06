import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure as IntFigure

from model.anomaly import Anomaly


def int_plot(
        title: str,
        df: pd.DataFrame) -> IntFigure:
    '''
    Produce a basic plot based on the DataFrame
    '''
    # set plotting backend
    pd.options.plotting.backend = "plotly"
    # base plot
    fig: IntFigure = df.plot()
    fig.layout.title = title
    return fig


def int_plot_marker(
        fig: IntFigure,
        xs: np.ndarray,
        ys: np.ndarray,
        color: str,
        name: str):
    '''
    Add markers to the fig obj
    color: css named colors
    '''
    markers = go.Scatter(
        x=xs, y=ys, mode='markers',
        marker=dict(color=color, size=10),
        name=name)
    fig.add_traces(markers)


def int_plot_color_region(
        fig: IntFigure,
        *,
        anomaly: Anomaly,
        width: int,
        annotation: str,
        color: str,
        opacity: float = 0.1,
        annotation_position='top left',
        layer='below'):
    '''
    Color a region of fig obj, from x to another x value
    anormaly: index postion of the anomaly
    width: total width of the colored region with anomaly at the center
    color: css named colors
    '''
    fig.add_vrect(
        x0=anomaly.idx - width / 2,
        x1=anomaly.idx + width / 2,
        fillcolor=color,
        opacity=opacity,
        annotation_text=annotation,
        annotation_position=annotation_position,
        layer=layer, line_width=0,
    )
