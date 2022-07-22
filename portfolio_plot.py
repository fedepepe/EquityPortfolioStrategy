#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 12:42:40 2021

@author: federico
"""
import logging

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from typing import Dict, Tuple

years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')


# Plot cumulative portfolio wealth
def plot_cum_wealth(cum_wealth_hist: pd.Series,
                    figure: plt.Figure = None,
                    descr: str = None,
                    linestyle: str = None
                    ) -> plt.Figure:
    x_data = cum_wealth_hist.index
    y_data = cum_wealth_hist.values

    # rnd_col = (np.random.rand(), np.random.rand(), np.random.rand())

    if descr is None:
        descr = 'Strategy'
    else:
        descr = f'Strategy: {descr})'

    if figure is None:
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)

        ax.plot(x_data, y_data, linestyle=linestyle, label=descr, markersize=5)
        # ax.set_xticks(cum_wealth_hist.index)
        # ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in x_data],
        #                    rotation=45, ha='right', fontsize=8)
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        ax.set_ylabel('Cumulative wealth', fontsize=12)
    else:
        ax = figure.axes[0]
        ax.plot(x_data, y_data, linestyle=linestyle, label=descr, markersize=5)

    return figure


# Plot market index
def plot_mkt_index(cum_wealth_hist: pd.Series,
                   mkt_idx_df: pd.DataFrame,
                   figure: plt.Figure
                   ) -> plt.Figure:
    endow = cum_wealth_hist[0]
    mkt_idx_df_plot = mkt_idx_df.loc[cum_wealth_hist.index[0]:]
    mkt_equity_line = mkt_idx_df_plot / mkt_idx_df_plot.iloc[0] * endow
    figure.axes[0].plot(mkt_idx_df_plot.index, mkt_equity_line, '--b', label='Market index')
    return figure


# Plot stock closing prices
def plot_stk_price(last_price_df: pd.DataFrame, tckr: str):
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.plot(last_price_df.index, last_price_df[tckr], '-s')
    ax.set_xticks(last_price_df.index)
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in last_price_df.index],
                       rotation=45, ha='right')
    ax.set_ylabel('Closing price', fontsize=12)
    figure.suptitle(tckr, fontsize=16)


# Plot cumulative portfolio wealth versus number of stocks
def plot_vs_nstock(results_df: pd.DataFrame,
                   label: str,
                   figure: plt.Figure = None,
                   descr: str = None
                   ) -> plt.Figure:
    x_plot = results_df['Nstk'].values
    y_plot = results_df[label].values

    rnd_col = (np.random.rand(), np.random.rand(), np.random.rand())

    if descr is None:
        descr = 'portfolio strategy'
    else:
        descr = f'portfolio strategy ({descr})'

    if figure is None:
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)

        ax.plot(x_plot, y_plot, '-s', color=rnd_col, label=descr, markersize=5)
        ax.set_xticks(results_df['Nstk'])
        ax.set_ylabel(label, fontsize=12)
    else:
        ax = figure.axes[0]
        ax.plot(x_plot, y_plot, '-s', color=rnd_col, label=descr, markersize=5)

    return figure


def make_grid(results_df: pd.DataFrame,
              label: str
              ) -> pd.DataFrame:
    # Build dataframe with results loaded from file
    n_obs_ar = np.sort(np.unique(results_df['Nobs'].astype(int)))  # Number of past observations
    n_reb_ar = np.sort(np.unique(results_df['Nreb'].astype(int)))  # Rebalancing interval

    df = pd.DataFrame(np.nan, index=n_reb_ar, columns=n_obs_ar)

    for index, row in results_df.iterrows():
        df.loc[int(row['Nreb']), int(row['Nobs'])] = results_df.loc[index, label]
    return df.astype(float)


def build_watermelon_colormap(scale_param: Tuple[float, float, float],
                              reverse: bool = False
                              ) -> Tuple[colors.LinearSegmentedColormap, float, float]:
    (v_max, v_min, v_wht) = scale_param

    def interp(x_1, y_1, x_2, y_2, x_3):
        a = (y_1 - y_2) / (x_1 - x_2)
        b = y_1 - a * x_1
        return a * x_3 + b

    if v_wht <= 0.5 * (v_min + v_max):
        c_max, c_min = (1., interp(v_max, 1., v_wht, 0.5, v_min))
    else:
        c_max, c_min = (interp(v_min, 0., v_wht, 0.5, v_max), 0.)

    def red_eq(x):
        assert 0 <= x <= 1
        if 0 <= x < 0.5:
            return 0.8 + 0.2 * x / 0.5
        else:
            return 2 - 2 * x

    def green_eq(x):
        assert 0 <= x <= 1
        if 0 <= x < 0.5:
            return 2 * x
        else:
            return 1.2 - 0.4 * x

    vrlo = red_eq(c_min)
    vrhi = red_eq(c_max)
    vglo = green_eq(c_min)
    vghi = green_eq(c_max)

    if c_min <= 0.5 <= c_max:
        # This dictionary defines the colormap with transition from red to green
        cp = interp(c_min, 0., c_max, 1., 0.5)
        red = ((0.0, vrlo, vrlo),  # set to 0.8 so its not too bright at 0
               (cp, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
               (1.0, vrhi, vrhi))  # no red at 1
        green = ((0.0, vglo, vglo),  # no green at 0
                 (cp, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                 (1.0, vghi, vghi))  # set to 0.8 so its not too bright at 1
        blue = ((0.0, vglo, vglo),  # no blue at 0
                (cp, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                (1.0, vrhi, vrhi))  # no blue at 1

    else:
        red = ((0.0, vrlo, vrlo),  # set to 0.8 so its not too bright at 0
               (1.0, vrhi, vrhi))  # no red at 1
        green = ((0.0, vglo, vglo),  # no green at 0
                 (1.0, vghi, vghi))  # set to 0.8 so its not too bright at 1
        blue = ((0.0, min(vrlo, vglo), min(vrlo, vglo)),  # no blue at 0
                (1.0, min(vrhi, vghi), min(vrhi, vghi)))  # no blue at 1

    # This dictionary defines the colormap with red or green shades only
    if reverse:
        c_dict = {'red': green, 'green': red, 'blue': blue}
    else:
        c_dict = {'red': red, 'green': green, 'blue': blue}

    # Create the colormap using the dictionary
    watermelon_cm = colors.LinearSegmentedColormap('GnRd', c_dict)
    return watermelon_cm, v_max, v_min


def plot_heatmap_single(df: pd.DataFrame,
                        scale_param: Tuple[float, float, float] = None,
                        metric_label: str = None,
                        figure: plt.Figure = None,
                        pos: int = 0,
                        descr: str = None,
                        fig_options: Dict = None,
                        reverse: bool = False
                        ) -> plt.Figure:
    """ Plot heatmap of portfolio performance metric versus n. of observations and rebalancing interval
    label   : parameter to be plotted
    val_wht : value corresponding to white color """

    if scale_param is None:
        scale_param = (df.to_numpy().max(), df.to_numpy().min(),
                       0.5 * (df.to_numpy().max() + df.to_numpy().min()))

    if fig_options is None:
        fig_options = {'show_x_label': True, 'show_y_label': True, 'show_colorbar': True}

    extent = [0.5, len(df.columns) + 0.5, 0.5, len(df.index) + 0.5]

    # Build watermelon colormap
    watermelon_cm, val_max, val_min = build_watermelon_colormap(scale_param=scale_param,
                                                                reverse=reverse)

    if figure is None:
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
    else:
        ax = figure.axes[pos]

    hm = ax.imshow(df.iloc[::-1], cmap=watermelon_cm, vmin=val_min, vmax=val_max, extent=extent)
    if fig_options['show_x_label']:
        ax.set_xlabel('Window length', fontsize=12)
    if fig_options['show_y_label']:
        ax.set_ylabel('Rebalancing\ninterval', fontsize=12)
    ax.set_xticks(np.arange(1, len(df.columns) + 1))
    ax.set_yticks(np.arange(1, len(df.index) + 1))
    ax.set_xticklabels([str(x) for x in df.columns.to_list()], fontsize=8, rotation=90)
    ax.set_yticklabels([str(y) for y in df.index.to_list()], fontsize=8)
    ax.set_title(descr, fontsize=8)
    if fig_options['show_colorbar']:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.2 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        figure.colorbar(hm, cax=cax)
    if metric_label is not None:
        figure.suptitle(metric_label, fontsize=12)
    figure.tight_layout()
    # figure.show()
    return figure


def plot_heatmap_mosaic(results_dict: Dict,
                        title: str = None,
                        reverse: bool = False
                        ) -> plt.Figure:
    # Get minimum and maximum value of the performance parameter
    v_max, v_min, v_wht = -np.inf, np.inf, None
    for key, df in results_dict.items():
        if 'mkt' in key:  # The value associate to white is that of the benchmark portfolio
            v_wht = df.iloc[0, 0]
        else:
            v_max = max(v_max, df.to_numpy().max())
            v_min = min(v_min, df.to_numpy().min())

    if v_wht is None:
        logging.warning('Could not identify market reference value.')
        v_wht = 0.5 * (v_max + v_min)

    scale_param = (v_max, v_min, v_wht)

    n_stk = list(results_dict.keys())[0][0]
    algos = list(set([k[1] for k in list(results_dict.keys())]))
    wght_mtds = list(set([k[2] for k in list(results_dict.keys()) if not k[2] == 'mkt']))

    n_cols = len(wght_mtds)
    n_rows = len(algos)
    figure, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    figure.set_size_inches(24, 13.5)

    for algo in algos:
        pos_0 = n_cols * algos.index(algo)
        for n, wm in enumerate(wght_mtds):
            if wm == wght_mtds[0]:
                fig_options = {'show_x_label': False, 'show_y_label': True, 'show_colorbar': True}
            elif wm == wght_mtds[-1]:
                fig_options = {'show_x_label': False, 'show_y_label': False, 'show_colorbar': True}
            else:
                fig_options = {'show_x_label': False, 'show_y_label': False, 'show_colorbar': True}
            if algo == algos[-1]:
                fig_options['show_x_label'] = True

            figure = plot_heatmap_single(df=results_dict[(n_stk, algo, wm)],
                                         scale_param=scale_param,
                                         figure=figure,
                                         pos=pos_0 + n,
                                         descr=f'{algo}, {wm}',
                                         fig_options=fig_options,
                                         reverse=reverse)

    figure.suptitle(f'{title} ({n_stk} stocks)', fontsize=12)
    figure.tight_layout(pad=0.0, h_pad=1.0, w_pad=1.0)
    figure.show()
    return figure
