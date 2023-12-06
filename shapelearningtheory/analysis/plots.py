from matplotlib import pyplot as plt
import seaborn as sns
from typing import Callable
# local imports
from .helpers import Table

def plot_column_wise(table: Table, plotfun, fig_params={}, plot_params={}):
    cols, _ = table.get_size()
    fig, ax = plt.subplots(cols, 1, **fig_params)
    for i, (col_name, column) in enumerate(table.get_columns()):
        plotfun(column, ax[i], **plot_params)
        ax[i].set_title(col_name)
    return fig

def plot_table(table: Table, plotfun, fig_params={}, plot_params={}):
    cols, rows = table.get_size()
    fig, ax = plt.subplots(cols, rows, **fig_params)
    for i, col in enumerate(table.col_names):
        for j, row in enumerate(table.row_names):
            plotfun(table[col, row], ax[i, j], **plot_params)
            if j == 0:
                ax[i, j].set_ylabel(col)
            if i == 0:
                ax[i, j].set_title(row)
    return fig

