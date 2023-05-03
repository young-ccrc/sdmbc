#! usr/bin/python
# Funtions for main_figure.py
# -----------------------------------------------------------------------------------------------------------------
# This is a script for a comparison of the models used.
# It starts by importing several necessary packages
# including NumPy, pandas, xarray, and matplotlib.
# It also imports some user-defined variables and functions from another Python file called "user_input".

# After importing the packages, the file defines several functions,
# including "cal_mean" and "cal_std" for calculating statistics
# and "assign_w", "cal_acor", and "cal_ccor" for performing specific calculations in 3D.
# These functions calculate the mean, standard deviation, auto-, and cross- correlation over different time periods
# for each of the atmospheric variables. The resulting figures are saved in the specified output path.

# The plotting functions use the matplotlib and Basemap packages.

# Written by Youngil(Young) Kim
# PhD Candidate
# Water Research Centre
# Climate Change Research Centre
# University of New South Wales
# 2023-04-10
# -----------------------------------------------------------------------------------------------------------------

# Load pacakges ===================================
import pytest
import numpy as np  # Arrays and matrix math
import pandas as pd  # DataFrames
import xarray as xr  # Xarray
import matplotlib.pyplot as plt  # Plotting
from user_input import *

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import cm

# from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
from matplotlib.offsetbox import AnchoredText
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy
from scipy.stats import ks_2samp

# Silence warnings
import warnings

warnings.simplefilter("ignore")

# Load pacakges end ================================


# Functions for main_figure =========================================
# functions for 3D --------------------------------------------------


# Set variable hus, ta, and w
def assign_w(ds):
    """
    Add the w variable (wind speed) to a given dataset 'ds'.

    """
    dsd = ds.resample(time="D").sum("time")
    ds_w = xr.apply_ufunc(np.hypot, ds["u"], ds["v"], output_dtypes=[ds["u"].dtype])
    dsd_w = ds_w.resample(time="D").sum("time")
    dsd = dsd.assign(w=dsd_w)
    dsd = dsd.drop(["u", "v"])
    return dsd


# Climatological mean
def cal_mean(ds):
    """
    Calculate the climatological mean of a given dataset ds over different time periods.

    """
    dsd = ds.groupby("time.dayofyear").mean("time").mean("dayofyear")
    dsm = ds.groupby("time.month").mean("time").mean("month")
    dss = ds.groupby("time.season").mean("time").mean("season")
    dsy = ds.groupby("time.year").mean("time").mean("year")
    return dsd, dsm, dss, dsy


# Standard deviation
def cal_std(ds):
    """
    Calculate the standard deviation of a given dataset ds over different time periods.

    """
    dsd = ds.std("time")
    dsm = ds.resample(time="M").mean("time").std("time")
    dss = ds.resample(time="QS-DEC").mean("time").std("time")
    dsy = ds.resample(time="Y").mean("time").std("time")
    return dsd, dsm, dss, dsy


# Lag1 auto-correlation
def cal_acor(ds, period, variable):  # period: M, QS-DEC, Y
    """
    Calculate the lag1 auto-correlation of given variables 'variable'
    in a given dataset 'ds' over a given time period 'period'.

    """
    if period != "D":
        ds = ds.resample(time=period).mean("time")
    temp = []
    cor_matrix = np.zeros([len(variable), len(ds["lat"]), len(ds["lon"])])
    for v in range(0, len(variable)):
        var = variable[v]
        for i in range(0, len(ds["lat"])):
            for j in range(0, len(ds["lon"])):
                temp = np.corrcoef(
                    ds[var][1:, i, j], ds[var][0 : (len(ds.time) - 1), i, j]
                )
                cor_matrix[v, i, j] = temp[0, 1]
    return cor_matrix


# Lag0 cross-correlation
def cal_ccor(ds, period, v):
    """
    Calculate the lag0 cross-correlation of given variables 'variable'
    in a given dataset 'ds' over a given time period 'period'.

    """
    if period != "D":
        ds = ds.resample(time=period).mean("time")
    temp = []
    cor_matrix = np.zeros([len(v), len(ds["lat"]), len(ds["lon"])])
    for i in range(0, len(ds["lat"])):
        for j in range(0, len(ds["lon"])):
            temp = np.corrcoef(ds[v[0]][:, i, j], ds[v[1]][:, i, j])
            cor_matrix[0, i, j] = temp[0, 1]
            temp = np.corrcoef(ds[v[0]][:, i, j], ds[v[2]][:, i, j])
            cor_matrix[1, i, j] = temp[0, 1]
            temp = np.corrcoef(ds[v[1]][:, i, j], ds[v[2]][:, i, j])
            cor_matrix[2, i, j] = temp[0, 1]
    return cor_matrix


# ks test
def ks_pvalue(a, b):
    statistic, pvalue = ks_2samp(a, b)
    return statistic, pvalue


def rounder(x):
    if x - int(x) >= 0.5:
        return int(np.ceil(x))
    else:
        return int(np.floor(x))


def calculate_ks_matrix(de, ds, variable):
    """
    Calculate the Kolmogorov-Smirnov (KS) test matrix and percentage of p-values above a given threshold (0.05) for each variable.

    Parameters:
    de (xarray.Dataset): Observed xarray dataset containing variables to be compared.
    ds (xarray.Dataset): Modelled xarray dataset containing variables to be compared.
    variable (list): List of variable names to perform the KS test on.

    Returns:
    ks_result (numpy.ndarray): An array containing the percentage of p-values greater than or equal to 0.05 for each variable.
    """

    ks_matrix = np.empty((len(variable), len(ds["lat"]), len(ds["lon"])))

    for k, var in enumerate(variable):
        g_statistic, g_pvalue = xr.apply_ufunc(
            ks_pvalue,
            de[var],
            ds[var],
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[[], []],
            vectorize=True,
        )

        ks_matrix[k, :, :] = g_pvalue

    ks_result = np.empty([len(variable)])

    for k in range(0, len(variable)):
        ks_out = xr.where(ks_matrix[k] >= 0.05, 1, 0)
        ks_result[k] = 100 * np.sum(ks_out) / (len(ds.lat) * len(ds.lon))

    return ks_result


# Plot functions
# For lag1 auto-correlation
def scatter_auto(ax, ds, de, markers, alpha, label, sz, v, limlist):
    """
    Draw scatter plots for lag1 auto-correlation

    """
    for m in range(0, 4):
        ax.scatter(
            de[m][v, :],
            ds[m][v, :],
            s=sz,
            marker=markers[m],
            label=label[m],
            alpha=alpha,
        )
    xpoints, ypoints = plt.xlim(limlist), plt.ylim(limlist)
    ax.plot(
        xpoints, ypoints, linestyle="--", color="k", lw=1, scalex=False, scaley=False
    )


# For climatological mean
def scatter_mean(ax, ds, de, markers, alpha, label, sz, v):
    """
    Draw scatter plots for mean

    """
    for m in range(0, 4):
        ax.scatter(
            de[m][v], ds[m][v], s=sz, marker=markers[m], label=label[m], alpha=alpha
        )
    xpoints, ypoints = plt.xlim(), plt.ylim()
    ax.plot(
        xpoints, ypoints, linestyle="--", color="k", lw=1, scalex=False, scaley=False
    )


# For standard deviation
def scatter_sd(ax, ds, de, markers, alpha, label, sz, v):
    """
    Draw scatter plots for standard deviation

    """
    for m in range(0, 4):
        ax.scatter(
            de[m][v], ds[m][v], s=sz, marker=markers[m], label=label[m], alpha=alpha
        )
    xpoints, ypoints = plt.xlim(), plt.ylim()
    ax.plot(
        xpoints, ypoints, linestyle="--", color="k", lw=1, scalex=False, scaley=False
    )


# For lag0 cross-correlation
def scatter_dot(ax, ds, de, markers, alpha, label, sz, v):
    """
    Draw scatter plots for lag0 cross-correlation

    """
    for m in range(0, 4):
        ax.scatter(
            de[m][v, :],
            ds[m][v, :],
            s=sz,
            marker=markers[m],
            label=label[m],
            alpha=alpha,
        )
    xpoints, ypoints = plt.xlim(), plt.ylim()
    ax.plot(
        xpoints, ypoints, linestyle="--", color="k", lw=1, scalex=False, scaley=False
    )


# Scatter plot of 3d atmospheric variables
def save_figure_3d(g_day, d_day, e_day, out_figure_path):
    """
    Save the scatter plots for three dimensional atmospheric variables
    including mean, standard deviation, lag1 auto-correlation.
    g_day: an array containing data from the raw GCM data
    d_day: an array containing data from the bias-corrected GCM data
    e_day: an array containing data from the observed data
    out_figure_path: a string representing the path where the output figure will be saved

    """
    # generate a list of markers and another of colors
    variable = ["q", "t", "w"]
    markers = ["s", "+", "^", "o"]
    colors = ["b", "g", "c", "r"]
    label = ["Day", "Month", "Season", "Year"]
    alpha = 0.3
    sz = 15
    limitlist = [-0.4, 1.0]

    print("Calculate statistics for 3D")
    # calculate mean over whole periods
    d_d, d_m, d_s, d_y = cal_mean(d_day)
    e_d, e_m, e_s, e_y = cal_mean(e_day)
    g_d, g_m, g_s, g_y = cal_mean(g_day)
    # calculate std over whole periods
    d_ds, d_ms, d_ss, d_ys = cal_std(d_day)
    e_ds, e_ms, e_ss, e_ys = cal_std(e_day)
    g_ds, g_ms, g_ss, g_ys = cal_std(g_day)

    # calculate auto-corr over whole periods
    d_da, d_ma, d_sa, d_ya = (
        cal_acor(d_day, "D", variable),
        cal_acor(d_day, "M", variable),
        cal_acor(d_day, "QS-DEC", variable),
        cal_acor(d_day, "Y", variable),
    )
    e_da, e_ma, e_sa, e_ya = (
        cal_acor(e_day, "D", variable),
        cal_acor(e_day, "M", variable),
        cal_acor(e_day, "QS-DEC", variable),
        cal_acor(e_day, "Y", variable),
    )
    g_da, g_ma, g_sa, g_ya = (
        cal_acor(g_day, "D", variable),
        cal_acor(g_day, "M", variable),
        cal_acor(g_day, "QS-DEC", variable),
        cal_acor(g_day, "Y", variable),
    )

    no_of_variables = len(variable)
    limlist = [-1.0, 1.0]

    # list  variables
    g_m_list = [g / 4 for g in [g_d, g_m, g_s, g_y]]
    g_sd_list = [g / 4 for g in [g_ds, g_ms, g_ss, g_ys]]
    g_auto_list = [g_da, g_ma, g_sa, g_ya]

    e_m_list = [e / 4 for e in [e_d, e_m, e_s, e_y]]
    e_sd_list = [e / 4 for e in [e_ds, e_ms, e_ss, e_ys]]
    e_auto_list = [e_da, e_ma, e_sa, e_ya]

    d_m_list = [d / 4 for d in [d_d, d_m, d_s, d_y]]
    d_sd_list = [d / 4 for d in [d_ds, d_ms, d_ss, d_ys]]
    d_auto_list = [d_da, d_ma, d_sa, d_ya]

    print("Save figures")

    for j in range(0, no_of_variables):
        fig = plt.figure(figsize=(18, 10))
        for i in range(1, 7):
            ax = fig.add_subplot(2, 3, i)
            if i == 1:
                scatter_mean(
                    ax, g_m_list, e_m_list, markers, alpha, label, sz, variable[j]
                )
                ax.legend()
                plt.ylabel("GCM", fontsize=18)
                plt.title("Mean", fontsize=20)
            elif i == 2:
                scatter_sd(
                    ax, g_sd_list, e_sd_list, markers, alpha, label, sz, variable[j]
                )
                plt.title("Standard Deviation", fontsize=20)
            elif i == 3:
                scatter_auto(
                    ax,
                    g_auto_list,
                    e_auto_list,
                    markers,
                    alpha,
                    label,
                    sz,
                    j,
                    limitlist,
                )
                plt.title("LAG1 correlation", fontsize=20)
            elif i == 4:
                scatter_mean(
                    ax, d_m_list, d_m_list, markers, alpha, label, sz, variable[j]
                )
                plt.ylabel("Bias-corrected GCM", fontsize=18)
                plt.xlabel("Obs", fontsize=18)
            elif i == 5:
                scatter_sd(
                    ax, d_sd_list, d_sd_list, markers, alpha, label, sz, variable[j]
                )
                plt.xlabel("Obs", fontsize=18)
            elif i == 6:
                scatter_auto(
                    ax,
                    d_auto_list,
                    e_auto_list,
                    markers,
                    alpha,
                    label,
                    sz,
                    j,
                    limitlist,
                )
                plt.xlabel("Obs", fontsize=18)

        fig.tight_layout()
        var_surface = "RCM"
        f_surface_name = "input.statistics"
        plt.savefig(
            f"{out_figure_path}{var_surface}.{f_surface_name}.{variable[j]}.png",
            dpi=100,
            bbox_inches="tight",
        )


# Scatter plot of 3d cross-correlation
def save_figure_3d_cross(g_day, d_day, e_day, out_figure_path):
    """
    Save the scatter plots for three dimensional atmospheric variables
    including lag0 cross-correlation.
    g_day: an array containing data from the raw GCM data
    d_day: an array containing data from the bias-corrected GCM data
    e_day: an array containing data from the observed data
    out_figure_path: a string representing the path where the output figure will be saved

    """
    # generate a list of markers and another of colors
    # generate a list of markers and another of colors
    variable = ["q", "t", "w"]
    markers = ["s", "+", "^", "o"]
    colors = ["black", "skyblue", "lightcoral", "silver"]
    cc = "viridis"
    label = ["Day", "Month", "Season", "Year"]
    alpha = 0.3
    sz = 15

    print("Calculate cross-correlation for 3D")
    # calculate cross-corr over whole periods
    d_dc, d_mc, d_sc, d_yc = (
        cal_ccor(d_day, "D", variable),
        cal_ccor(d_day, "M", variable),
        cal_ccor(d_day, "QS-DEC", variable),
        cal_ccor(d_day, "Y", variable),
    )
    e_dc, e_mc, e_sc, e_yc = (
        cal_ccor(e_day, "D", variable),
        cal_ccor(e_day, "M", variable),
        cal_ccor(e_day, "QS-DEC", variable),
        cal_ccor(e_day, "Y", variable),
    )
    g_dc, g_mc, g_sc, g_yc = (
        cal_ccor(g_day, "D", variable),
        cal_ccor(g_day, "M", variable),
        cal_ccor(g_day, "QS-DEC", variable),
        cal_ccor(g_day, "Y", variable),
    )

    # list  variables
    g_c_list = [g_dc, g_mc, g_sc, g_yc]
    e_c_list = [e_dc, e_mc, e_sc, e_yc]
    d_c_list = [d_dc, d_mc, d_sc, d_yc]

    # Cross-correlation titles
    ctitle = ["q & T", "q & w", "T & w"]
    print("Save figures")

    fig = plt.figure(figsize=(18, 10))
    for i in range(1, 7):
        ax = fig.add_subplot(2, 3, i)
        if i < 4:
            scatter_dot(ax, g_c_list, e_c_list, markers, alpha, label, sz, i - 1)
            plt.title(ctitle[i - 1], fontsize=20)
            plt.xticks([])
            if i > 1:
                plt.yticks([])
            if i == 1:
                plt.ylabel("GCM", fontsize=18)
                ax.legend()
        elif i > 3 and i < 7:
            scatter_dot(ax, d_c_list, e_c_list, markers, alpha, label, sz, i - 4)
            plt.xticks([])
            if i > 4:
                plt.yticks([])
            if i == 4:
                plt.ylabel("Bias-corrected GCM", fontsize=18)
                plt.xlabel("Obs", fontsize=18)
    fig.tight_layout()
    var_surface = "RCM"
    f_surface_name = "input.statistics"
    plt.savefig(
        f"{out_figure_path}{var_surface}.{f_surface_name}.cross_correlation.png",
        dpi=100,
        bbox_inches="tight",
    )


# functions for 3D end ----------------------------------------------

# functions for sst -------------------------------------------------


# Auto-correlation for sst
def cal_acor_sst(ds, period):  # period: M, QS-DEC, Y
    """
    Calculate the lag1 auto-correlation of given surface variable 'variable'
    in a given dataset 'ds' over a given time period 'period'.

    """
    if period != "D":
        ds = ds.resample(time=period).mean("time")
    temp = []
    cor_matrix = np.zeros([len(ds.lat), len(ds.lon)])
    var = "sst"
    for i in range(0, len(ds.lat)):
        for j in range(0, len(ds.lon)):
            temp = np.corrcoef(ds[var][1:, i, j], ds[var][:-1, i, j])
            cor_matrix[i, j] = temp[0, 1]
    return cor_matrix


# Contour plot of sst
def save_figure_surface(g_sst, d_sst, e_sst, out_figure_path):
    """
    Save the contourf plots for two dimensional surface variable (sst)
    including mean absolute error.
    g_sst: an array containing data from the raw GCM data
    d_sst: an array containing data from the bias-corrected GCM data
    e_sst: an array containing data from the observed data
    out_figure_path: a string representing the path where the output figure will be saved

    """
    variable = "sst"
    # Statistics calculation =======================
    print("Calculate statistics")
    # calculate mean over whole periods
    d_d, d_m, d_s, d_y = cal_mean(d_sst)
    e_d, e_m, e_s, e_y = cal_mean(e_sst)
    g_d, g_m, g_s, g_y = cal_mean(g_sst)

    # calculate std over whole periods
    d_ds, d_ms, d_ss, d_ys = cal_std(d_sst)
    e_ds, e_ms, e_ss, e_ys = cal_std(e_sst)
    g_ds, g_ms, g_ss, g_ys = cal_std(g_sst)

    # calculate auto-corr over whole periods
    d_da, d_ma, d_sa, d_ya = (
        cal_acor_sst(d_sst, "D"),
        cal_acor_sst(d_sst, "M"),
        cal_acor_sst(d_sst, "QS-DEC"),
        cal_acor_sst(d_sst, "Y"),
    )
    e_da, e_ma, e_sa, e_ya = (
        cal_acor_sst(e_sst, "D"),
        cal_acor_sst(e_sst, "M"),
        cal_acor_sst(e_sst, "QS-DEC"),
        cal_acor_sst(e_sst, "Y"),
    )
    g_da, g_ma, g_sa, g_ya = (
        cal_acor_sst(g_sst, "D"),
        cal_acor_sst(g_sst, "M"),
        cal_acor_sst(g_sst, "QS-DEC"),
        cal_acor_sst(g_sst, "Y"),
    )

    # nan where e is zero
    g_ss = xr.where(e_ss == 0, np.nan, g_ss)
    d_ss = xr.where(e_ss == 0, np.nan, d_ss)
    e_ss = xr.where(g_ss == 0, np.nan, e_ss)

    g_s = xr.where(e_s == 0, np.nan, g_s)
    d_s = xr.where(e_s == 0, np.nan, d_s)
    e_s = xr.where(g_s == 0, np.nan, e_s)

    gss_bias = g_ss.sst[5:-5, 5:-5] - e_ss.sst[5:-5, 5:-5]
    dss_bias = d_ss.sst[5:-5, 5:-5] - e_ss.sst[5:-5, 5:-5]

    gsm_bias = g_s.sst[5:-5, 5:-5] - e_s.sst[5:-5, 5:-5]
    dsm_bias = d_s.sst[5:-5, 5:-5] - e_s.sst[5:-5, 5:-5]

    gsa_bias = g_sa[5:-5, 5:-5] - e_sa[5:-5, 5:-5]
    dsa_bias = d_sa[5:-5, 5:-5] - e_sa[5:-5, 5:-5]

    gss_biasm = np.nanmean(abs(gss_bias))
    dss_biasm = np.nanmean(abs(dss_bias))

    gsm_biasm = np.nanmean(abs(gsm_bias))
    dsm_biasm = np.nanmean(abs(dsm_bias))

    gsa_biasm = np.nanmean(abs(gsa_bias))
    dsa_biasm = np.nanmean(abs(dsa_bias))

    # inputs
    datasets = [gsm_bias, gss_bias, gsa_bias, dsm_bias, dss_bias, dsa_bias]

    model_bias = [gsm_biasm, gss_biasm, gsa_biasm, dsm_biasm, dss_biasm, dsa_biasm]

    # col, row
    columns = 3
    rows = 2

    # contour levels and ticks
    mlevels = np.linspace(-2.0, 2.0, 61, endpoint=True, dtype=float)

    slevels = np.linspace(-0.5, 0.5, 61, endpoint=True, dtype=float)

    alevels = np.linspace(-0.2, 0.2, 61, endpoint=True, dtype=float)

    ticks = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    fsize = 15

    # coords
    lat = e_s["lat"][5:-5].values
    lon = e_s["lon"][5:-5].values
    titles = ["GCM", "Bias-corrected GCM"]
    title = ["Seasonal M", "Seasonal Std.", "Seasonal Lag1"]

    print("Save figures")

    # draw figure
    fig = plt.figure(figsize=(12, 6))
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i, projection=ccrs.PlateCarree(180))
        if i in [1, 4]:
            levels = mlevels
        elif i in [2, 5]:
            levels = slevels
        else:
            levels = alevels

        # filled contours
        cf = ax.contourf(
            lon,
            lat,
            datasets[i - 1],
            levels=levels,
            cmap="RdBu",
            transform=ccrs.PlateCarree(),
            extend="both",
        )

        plt.gca().set_facecolor("silver")

        # set backgrounds
        ax.coastlines()
        ax.add_feature(cartopy.feature.LAND, edgecolor="black")

        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.2,
            color="gray",
            alpha=0.7,
            linestyle="--",
            dms=True,
        )

        # add mean value
        at = AnchoredText(
            np.round(model_bias[i - 1], 1),
            prop=dict(size=15),
            frameon=True,
            loc="lower right",
        )

        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

        if i < 4:
            if i == 1:
                # plt.ylabel(titles[0]+'\n ',fontsize=20)
                plt.ylabel(titles[0] + "\n ", fontsize=fsize, multialignment="center")
                plt.title(title[i - 1] + "\n ", fontsize=fsize)
                gl.top_labels = False
                gl.right_labels = False
                gl.bottom_labels = False
                # plt.gca().axes.get_xaxis().set_visible(True)
                plt.gca().axes.get_yaxis().set_visible(True)
                ax.set_yticklabels([])
            else:
                plt.title(title[i - 1] + "\n ", fontsize=fsize)
                gl.top_labels = False
                gl.left_labels = False
                gl.right_labels = False
                gl.bottom_labels = False
        if i > 3 and i < 7:
            if i == 4:
                plt.ylabel(titles[1] + "\n ", fontsize=fsize, multialignment="center")
                gl.top_labels = False
                gl.right_labels = False
                # gl.bottom_labels = False
                plt.gca().axes.get_yaxis().set_visible(True)
                ax.set_yticklabels([])
            else:
                gl.top_labels = False
                gl.right_labels = False
                gl.left_labels = False
                gl.bottom_labels = False

        if i == 4:
            # color bar location [left, bottom, width, height]
            cbaxes = fig.add_axes([0.1255, 0.1, 0.255, 0.01])
            cbar = fig.colorbar(
                cf,
                cax=cbaxes,
                ticks=[-2.0, -1.0, 0, 1.0, 2.0],
                orientation="horizontal",
                extend="both",
            )
            cbar.set_ticklabels([-2.0, -1.0, 0, 1.0, 2.0])
        elif i == 5:
            cbaxes = fig.add_axes([0.3855, 0.1, 0.255, 0.01])
            cbar = fig.colorbar(
                cf,
                cax=cbaxes,
                ticks=[-0.5, -0.25, 0, 0.25, 0.5],
                orientation="horizontal",
                extend="both",
            )
            cbar.set_ticklabels([-0.5, -0.25, 0, 0.25, 0.5])

        elif i == 6:
            cbaxes = fig.add_axes([0.6455, 0.1, 0.255, 0.01])
            cbar = fig.colorbar(
                cf,
                cax=cbaxes,
                ticks=[-0.2, -0.1, 0, 0.1, 0.2],
                orientation="horizontal",
                extend="both",
            )
            cbar.set_ticklabels([-0.2, -0.1, 0, 0.1, 0.2])

    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    var_surface = "sst"
    f_surface_name = "seasonal_stats_all"
    plt.savefig(
        f"{out_figure_path}{var_surface}_{f_surface_name}.png",
        dpi=100,
        bbox_inches="tight",
    )


# functions for sst end ---------------------------------------------

# functions for ks-test ---------------------------------------------
# Added in the main script

# Functions for main_figure end =====================================
