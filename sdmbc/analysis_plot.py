#! usr/bin/python
# =======================================================================================
# Description
#
# Input file formats
# - Atmospheric fields: 3D.%model%.%idx%.input.nc
# - Surface fields: sfc.%model%.input.nc

# This script is to draw the output figures of bias-corrected GCM with regard to Obs.
# This includes
# 1. Scatter plot of three statistics: mean, standard deviation, lag1 auto-correlation
# 2. Scatter plot of cross-correlation between atmospheric variables
# 3. Contour plot of surface vairable: sea surface temperature (SST)
# 4. The Kolmogorov-Smirnov (K-S) test result for each variable: hus, ta, w

# One of the grid cells within a speficied domain will be chosen to test the output.

# Written by Youngil(Young) Kim
# PhD Candidate
# Water Research Centre
# Climate Change Research Centre
# University of New South Wales
# 2023-04-17
# =======================================================================================
import pytest
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import ks_2samp

from user_input import *
from figurefunction import (
    assign_w,
    save_figure_3d,
    save_figure_3d_cross,
    save_figure_surface,
    rounder,
    calculate_ks_matrix,
)


class AnalysisBC:
    """
    A class to analyze bias-corrected climate model data.

    Attributes:
    -----------
    bc_path : str
        Path to the folder containing the bias-corrected data.
    lat_range : tuple
        A tuple containing the range of latitude indices to analyze.
    lon_range : tuple
        A tuple containing the range of longitude indices to analyze.
    startyear : int
        The starting year of the data.
    out_figure_path : str
        The path to save output figures.
    kstest : bool, optional
        If True, the Kolmogorov-Smirnov (K-S) test will be performed,
        otherwise not. Default is False.
    """

    def __init__(
        self, bc_path, lat_range, lon_range, startyear, out_figure_path, kstest=False
    ):
        self.bc_path = bc_path
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.startyear = startyear
        self.out_figure_path = out_figure_path
        self.kstest = kstest

    def figure_atmos(self):
        """
        Analyzes atmospheric variables and generates plots for bias-corrected data.
        Saves scatter plots of mean, standard deviation, and lag1 auto-correlation
        for the atmospheric variables, as well as cross-correlation between them.
        Optionally, it can also perform the Kolmogorov-Smirnov (K-S) test
        and print the results.
        """

        input_gcm_3d = "3d.gcm.1.input.nc"
        input_obs_3d = "3d.obs.1.input.nc"
        input_bcd_3d = "3d.bcd.1.output.nc"

        self.dgcm_3d = xr.open_dataset(f"{self.bc_path}{input_gcm_3d}")
        self.dobs_3d = xr.open_dataset(f"{self.bc_path}{input_obs_3d}")
        self.dbcd_3d = xr.open_dataset(f"{self.bc_path}{input_bcd_3d}")

        # Reformat time
        times = pd.date_range(
            "%s-01-01" % (self.startyear), freq="6H", periods=len(self.dgcm_3d.time)
        )

        self.dgcm_3d = self.dgcm_3d.update({"time": times})
        self.dobs_3d = self.dobs_3d.update({"time": times})
        self.dbcd_3d = self.dbcd_3d.update({"time": times})

        # Select a first grid cell to test 3D output
        self.g = self.dgcm_3d.isel(
            lat=slice(self.lat_range[0] + 1, self.lat_range[1] - 1),
            lon=slice(self.lon_range[0] + 1, self.lon_range[1] - 1),
        )
        self.e = self.dobs_3d.isel(
            lat=slice(self.lat_range[0] + 1, self.lat_range[1] - 1),
            lon=slice(self.lon_range[0] + 1, self.lon_range[1] - 1),
        )
        self.d = self.dbcd_3d.isel(
            lat=slice(self.lat_range[0] + 1, self.lat_range[1] - 1),
            lon=slice(self.lon_range[0] + 1, self.lon_range[1] - 1),
        )

        # Multiply 1000 to q to avoid a small value
        self.g["q"] *= 1000
        self.e["q"] *= 1000
        self.d["q"] *= 1000

        # resample from 6-hourly to daily
        self.g_day = assign_w(self.g)
        self.e_day = assign_w(self.e)
        self.d_day = assign_w(self.d)

        # Save 1. Scatter plot of three statistics: mean, standard deviation, lag1 auto-correlation
        save_figure_3d(self.g_day, self.d_day, self.e_day, self.out_figure_path)

        # Save 2. Scatter plot of cross-correlation between atmospheric variables
        save_figure_3d_cross(self.g_day, self.d_day, self.e_day, self.out_figure_path)

        if self.kstest == True:
            variable = ["q", "t", "w"]
            # Save 4. The Kolmogorov-Smirnov (K-S) test result for each variable: hus, ta, w
            # Perform KS test on the two samples
            self.gks_result = calculate_ks_matrix(self.e_day, self.g_day, variable)
            self.dks_result = calculate_ks_matrix(self.e_day, self.d_day, variable)

            for j in range(0, len(variable)):
                var = variable[j]
                print("")
                print(
                    f"Kolmogorov-Smirnov (K-S) test over the domain for the variable: {var}"
                )
                # print(f"For variable {var}")
                print("")
                print(f"• KS test for raw GCM")
                print(
                    f" - {rounder(self.gks_result[j])}% of the raw GCM are likely to be drawn from the same distribution."
                )
                print("")
                print(f"• KS test for bias-corrected GCM")
                print(
                    f" - {rounder(self.dks_result[j])}% of the bias-corrected GCM are likely to be drawn from the same distribution."
                )
                print("")
                print("• Result")
                print(
                    f" - The bias-corrected GCM presents {rounder(self.dks_result[j])-rounder(self.gks_result[j])}% improvement compared to the raw GCM."
                )
                print("")

    def figure_surface(self):
        """
        Analyzes surface variables and generates plots for bias-corrected data.
        Saves contour plots of sea surface temperature (SST) for
        raw GCM, bias-corrected GCM, and observation data.
        """

        input_gcm_sst = "sfc.gcm.input.nc"
        input_obs_sst = "sfc.obs.input.nc"
        input_bcd_sst = "sfc.bcd.output.nc"

        self.dgcm_sst = xr.open_dataset(f"{self.bc_path}{input_gcm_sst}")
        self.dobs_sst = xr.open_dataset(f"{self.bc_path}{input_obs_sst}")
        self.dbcd_sst = xr.open_dataset(f"{self.bc_path}{input_bcd_sst}")

        # Reformat time
        times = pd.date_range(
            "%s-01-01" % (self.startyear), freq="6H", periods=len(self.dgcm_3d.time)
        )

        self.dgcm_sst = self.dgcm_sst.update({"time": times})
        self.dobs_sst = self.dobs_sst.update({"time": times})
        self.dbcd_sst = self.dbcd_sst.update({"time": times})

        # Add coordinates
        self.dbcd_sst["lat"], self.dbcd_sst["lon"] = (
            self.dgcm_sst.lat,
            self.dgcm_sst.lon,
        )

        # Select a first grid cell to test sst output
        self.g_sst = self.dgcm_sst.isel(
            lat=slice(self.lat_range[0], self.lat_range[1]),
            lon=slice(self.lon_range[0], self.lon_range[1]),
        )
        self.e_sst = self.dobs_sst.isel(
            lat=slice(self.lat_range[0], self.lat_range[1]),
            lon=slice(self.lon_range[0], self.lon_range[1]),
        )
        self.d_sst = self.dbcd_sst.isel(
            lat=slice(self.lat_range[0], self.lat_range[1]),
            lon=slice(self.lon_range[0], self.lon_range[1]),
        )

        # Replace sst values greater than 1000 or equal to 0 with NaN
        self.g_sst = xr.where(
            (self.g_sst > 1000) | (self.g_sst == 0), np.nan, self.g_sst
        )
        self.e_sst = xr.where(
            (self.e_sst > 1000) | (self.e_sst == 0), np.nan, self.e_sst
        )
        self.d_sst = xr.where(
            (self.d_sst > 1000) | (self.d_sst == 0), np.nan, self.d_sst
        )

        self.g_sst = self.g_sst.resample(time="D").sum("time")
        self.e_sst = self.e_sst.resample(time="D").sum("time")
        self.d_sst = self.d_sst.resample(time="D").sum("time")

        # Save 3. Contour plot of surface vairable: sea surface temperature (SST)
        save_figure_surface(self.g_sst, self.d_sst, self.e_sst, self.out_figure_path)
