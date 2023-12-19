from matplotlib import pyplot as plt
import rsatoolbox
import seaborn as sns
from typing import Callable, List
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

def plot_rdm_grid(rdm_list: List[rsatoolbox.rdm.RDMs]):
    rows = len(rdm_list)
    cols = max(len(rdms.dissimilarities) for rdms in rdm_list)
    fig, ax = plt.subplots(rows, cols)
    for row, rdms in enumerate(rdm_list):
        for col, rdm in enumerate(rdms):
            ax[row][col].imshow(
                rdm.get_matrices().squeeze(),
                cmap="bone",
                vmin=0.0,
                interpolation="none")
            # set column titles
            ax[row][col].set_title(rdm.rdm_descriptors["property"][0])
            ax[row][col].set_axis_off()
        for col in range(len(rdms), cols):
            ax[row][col].set_axis_off()
        # set row titles
        ax[row][0].get_yaxis().set_visible(True)
        ax[row][0].set_ylabel(rdms.rdm_descriptors["feature_type"][0])
    return fig
