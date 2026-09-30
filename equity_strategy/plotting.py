"""Plots of portfolio wealth and of parameter-sweep heatmaps."""
import logging

import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from equity_strategy.backtest.results import GridKey

logger = logging.getLogger(__name__)

ScaleParams = tuple[float, float, float]  # (max value, min value, value mapped to white)


def plot_cum_wealth(value_history: pd.Series,
                    figure: plt.Figure | None = None,
                    description: str | None = None,
                    linestyle: str | None = None) -> plt.Figure:
    """Plot a wealth curve, on a new figure or on the first axes of ``figure``."""
    label = "Strategy" if description is None else f"Strategy: {description}"

    if figure is None:
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.set_ylabel("Cumulative wealth", fontsize=12)
    else:
        ax = figure.axes[0]

    ax.plot(value_history.index, value_history.values, linestyle=linestyle, label=label, markersize=5)
    return figure


def _interpolate(x_1: float, y_1: float, x_2: float, y_2: float, x: float) -> float:
    slope = (y_1 - y_2) / (x_1 - x_2)
    return slope * x + y_1 - slope * x_1


def _red(x: float) -> float:
    return 0.8 + 0.2 * x / 0.5 if x < 0.5 else 2 - 2 * x


def _green(x: float) -> float:
    return 2 * x if x < 0.5 else 1.2 - 0.4 * x


def build_watermelon_colormap(scale: ScaleParams,
                              reverse: bool = False) -> tuple[colors.LinearSegmentedColormap, float, float]:
    """Red-white-green colormap with white at the reference value (green = better unless ``reverse``)."""
    v_max, v_min, v_white = scale

    # position of the data range on a [0, 1] scale where the reference value sits at 0.5
    if v_white <= 0.5 * (v_min + v_max):
        c_max, c_min = 1.0, _interpolate(v_max, 1.0, v_white, 0.5, v_min)
    else:
        c_max, c_min = _interpolate(v_min, 0.0, v_white, 0.5, v_max), 0.0
    red_lo, red_hi = _red(c_min), _red(c_max)
    green_lo, green_hi = _green(c_min), _green(c_max)

    if c_min <= 0.5 <= c_max:  # red, through white, to green
        white_at = _interpolate(c_min, 0.0, c_max, 1.0, 0.5)
        red = ((0.0, red_lo, red_lo), (white_at, 1.0, 1.0), (1.0, red_hi, red_hi))
        green = ((0.0, green_lo, green_lo), (white_at, 1.0, 1.0), (1.0, green_hi, green_hi))
        blue = ((0.0, green_lo, green_lo), (white_at, 1.0, 1.0), (1.0, red_hi, red_hi))
    else:  # shades of red only, or of green only
        red = ((0.0, red_lo, red_lo), (1.0, red_hi, red_hi))
        green = ((0.0, green_lo, green_lo), (1.0, green_hi, green_hi))
        blue_lo, blue_hi = min(red_lo, green_lo), min(red_hi, green_hi)
        blue = ((0.0, blue_lo, blue_lo), (1.0, blue_hi, blue_hi))

    channels = {"red": green, "green": red, "blue": blue} if reverse else {"red": red, "green": green, "blue": blue}
    return colors.LinearSegmentedColormap("GnRd", channels), v_max, v_min


def plot_heatmap_single(grid: pd.DataFrame,
                        ax: plt.Axes,
                        scale: ScaleParams,
                        title: str | None = None,
                        show_x_label: bool = True,
                        show_y_label: bool = True,
                        show_colorbar: bool = True,
                        reverse: bool = False) -> None:
    """Heatmap of a metric versus window length (columns) and rebalancing interval (rows)."""
    colormap, v_max, v_min = build_watermelon_colormap(scale, reverse=reverse)
    extent = [0.5, len(grid.columns) + 0.5, 0.5, len(grid.index) + 0.5]

    image = ax.imshow(grid.iloc[::-1], cmap=colormap, vmin=v_min, vmax=v_max, extent=extent)
    if show_x_label:
        ax.set_xlabel("Window length", fontsize=12)
    if show_y_label:
        ax.set_ylabel("Rebalancing\ninterval", fontsize=12)
    ax.set_xticks(np.arange(1, len(grid.columns) + 1))
    ax.set_yticks(np.arange(1, len(grid.index) + 1))
    ax.set_xticklabels([str(x) for x in grid.columns], fontsize=8, rotation=90)
    ax.set_yticklabels([str(y) for y in grid.index], fontsize=8)
    ax.set_title(title, fontsize=8)
    if show_colorbar:
        colorbar_ax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.2)
        ax.figure.colorbar(image, cax=colorbar_ax)


def plot_heatmap_mosaic(grids: dict[GridKey, pd.DataFrame],
                        title: str | None = None,
                        reverse: bool = False) -> plt.Figure:
    """One heatmap per (algorithm, weight method) for a single number of stocks.

    The market portfolio's value is mapped to white, so colors show over/under-performance.
    """
    v_max, v_min, v_white = -np.inf, np.inf, None
    for (_, _, method), grid in grids.items():
        if method == "mkt":
            v_white = grid.iloc[0, 0]
        else:
            v_max = max(v_max, np.nanmax(grid.to_numpy()))
            v_min = min(v_min, np.nanmin(grid.to_numpy()))
    if v_white is None:
        logger.warning("Could not identify the market reference value.")
        v_white = 0.5 * (v_max + v_min)
    scale = (v_max, v_min, v_white)

    n_stocks = next(iter(grids))[0]
    algorithms = sorted({algorithm for _, algorithm, _ in grids})
    methods = sorted({method for _, _, method in grids if method != "mkt"})

    figure, axes = plt.subplots(len(algorithms), len(methods), sharex=True, sharey=True, squeeze=False)
    figure.set_size_inches(24, 13.5)
    for row, algorithm in enumerate(algorithms):
        for col, method in enumerate(methods):
            plot_heatmap_single(grids[(n_stocks, algorithm, method)], axes[row, col], scale,
                                title=f"{algorithm}, {method}",
                                show_x_label=(row == len(algorithms) - 1),
                                show_y_label=(col == 0),
                                reverse=reverse)

    figure.suptitle(f"{title} ({n_stocks} stocks)", fontsize=12)
    figure.tight_layout(pad=0.0, h_pad=1.0, w_pad=1.0)
    return figure
