#! usr/bin/python
# Funtions for main_reformat.py
# -----------------------------------------------------------------------------------------------------------------
# This is a Python script named "inputfunction.py" that contains a set of functions
# to be used in another script named "main_reformat.py".

# The script imports the following modules:
# "user_input", "numpy", "pandas", "xarray", and "glob".
# It defines several functions that will be used in the "main_reformat.py" script
# to perform some data processing tasks.
#
# The functions are:
#
# add_lev_dim(new): This function adds a new dimension named "lev" to a netCDF file.
#
# copyenv(new, old, vn, vo): This function copies the environment of 3D variables (latitude, longitude, and time)
# from an old netCDF file to a new one, as well as the attributes and encoding of the variable of interest.
#
# reformatsave(bcf, var_new, var_old, y): This function is the main function for reformatting and saving output
# as netCDF format. It opens raw GCM datasets, renames variable names, concatenates if needed,
# transposes dimensions, creates a new dataset, copies attributes, changes values, and encodes the output.

# Written by Youngil(Young) Kim
# PhD Candidate
# Water Research Centre
# Climate Change Research Centre
# University of New South Wales
# 2023-04-17
# -----------------------------------------------------------------------------------------------------------------

# Load pacakges ===================================
import pytest
from user_input import *
import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
from functools import partial

# Load pacakges end ================================
# ---------------------------------------------------------------------------------------------------
# Functions to be used for main_reformat


def add_lev_dim(new):
    """
    Add levels to the bias-corrected netcdf files

    """

    new = new.expand_dims(lev=1)
    return new


def copyenv(new, old, vn, vo):
    """
    Copies the environment of 3D variables from an old netCDF file to a new one,
    along with variable attributes and encoding.
    Also adds level dimension if the variable is not sea surface temperature.

    """

    # copy enviornment of 3D variables
    for dim in ["lat", "lon", "time"]:
        new[dim] = old[dim]
        new[dim].attrs = old[dim].attrs
        new[dim].encoding = old[dim].encoding
        new[dim].encoding["_FillValue"] = None

    new[vn].attrs = old[vo].attrs
    new[vn].encoding = old[vo].encoding
    new[vn].encoding["_FillValue"] = None

    # if not sea surface temperature
    if vn != "tos":
        new["lev"] = old["lev"]
        new["lev"].attrs = old["lev"].attrs
        new["lev"].encoding = old["lev"].encoding
        new.lev.encoding["_FillValue"] = None

    # others
    new.attrs = old.attrs

    return new


def reformatsave_3d(bcf, var_new, var_old, y):
    """
    Reformat and save a 3D variable from bias-corrected data.

    Parameters:
    bcf (xarray.Dataset): an array contains bias-corrected GCM data over the research periods.
    var_new (str): Bias-corrected variables's name.
    var_old (str): Original GCM variable's name.
    y (int): The current year being processed.

    Returns:
    xarray.Dataset: Reformatted dataset with the updated 2D variable.
    """

    # main fucntion for reformatting and saving output as netcdf format.
    #

    # set years
    year = str(y)
    nyear = y + 1
    nyear = str(nyear)

    # open raw GCM datasets
    with xr.open_dataset(
        f"{gcm_path}{var_old}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{year}01010600-{nyear}01010000.nc"
    ) as esmf, xr.set_options(keep_attrs=True):
        # rename variable's name
        bcf_rn = bcf.rename({var_new: var_old})
        # concatenate if needs
        if y == endyear:  # last year concatenate
            xr_conf = xr.concat(
                [bcf_rn[var_old], esmf[var_old].isel(time=slice(-1, None))], dim="time"
            )
            bcf_sel = xr_conf.sel(
                time=slice("%s-01-01 06" % (year), "%s-01-01 00" % (nyear))
            )
            # transpose dims
            bcf_sel = bcf_sel.transpose("time", "lev", "lat", "lon")
            # create new dataset
            bcf_sel = xr.Dataset({"%s" % (var_old): bcf_sel})
            # copy attributes
            bcf_sel = copyenv(bcf_sel, esmf, var_old, var_old)
            # change values
            esmf[var_old] = bcf_sel[var_old]

        elif y == startyear - 1:  # previous year concatenate
            bcf_1 = bcf.isel(time=slice(None, 1))  # exclude first time step, mbcf
            bcf_rn = bcf_1.rename({var_new: var_old})
            bcf_sel = xr.concat(
                [
                    esmf[var_old].sel(
                        time=slice("%s-01-01 06" % (year), "%s-12-31 18" % (year))
                    ),
                    bcf_rn[var_old],
                ],
                dim="time",
            )
            # transpose dims
            bcf_sel = bcf_sel.transpose("time", "lev", "lat", "lon")
            # create new dataset
            bcf_sel = xr.Dataset({"%s" % (var_old): bcf_sel})
            # copy attributes
            bcf_sel = copyenv(bcf_sel, esmf, var_old, var_old)
            # change values
            esmf[var_old] = bcf_sel[var_old]

        else:
            bcf_sel = bcf_rn.sel(
                time=slice("%s-01-01 06" % (year), "%s-01-01 00" % (nyear))
            )
            # transpose dims
            bcf_sel = bcf_sel.transpose("time", "lev", "lat", "lon")
            # copy attributes
            bcf_sel = copyenv(bcf_sel, esmf, var_old, var_old)
            # change values
            esmf[var_old] = bcf_sel[var_old]

        # None encoding
        # for meridional wind vector
        if var_old == "va":
            for var in ["lev_bnds", "b", "orog", "b_bnds"]:
                esmf[var].encoding["_FillValue"] = None

        # for other variables
        else:
            for var in ["lev_bnds", "lat_bnds", "lon_bnds", "b", "orog", "b_bnds"]:
                esmf[var].encoding["_FillValue"] = None

    # return reformatted dataset
    return esmf


def reformatsave_2d(bcf, var_new, var_old, y):
    """
    Reformat and save a 2D variable from bias-corrected data.

    Parameters:
    bcf (xarray.Dataset): an array contains bias-corrected GCM data over the research periods.
    var_new (str): Bias-corrected variables's name.
    var_old (str): Original GCM variable's name.
    y (int): The current year being processed.

    Returns:
    xarray.Dataset: Reformatted dataset with the updated 2D variable.
    """

    var_new = "sst"
    var_old = "tos"
    # starty_sfc=2010
    #    for y in range(starty_sfc,endy_sfc+1,10):
    year = str(y)
    nyear = y + 9
    if nyear > 2009:
        nyear = y + 4
    nyear = str(nyear)

    with xr.open_dataset(
        f"{gcm_path}{var_old}_Oday_{gname}_{period}_{cinfor}_{sinfor}_{year}0101-{nyear}1231.nc"
    ) as esmf, xr.set_options(keep_attrs=True):
        # rename variable's name
        sfcfd_rn = bcf.rename({var_new: var_old})
        # concatenate if needs
        if y == starty_sfc:
            xr_conf = xr.concat(
                [
                    esmf[var_old].sel(
                        time=slice(
                            "%s-01-01" % (starty_sfc), "%s-12-31" % (starty_sfc + 1)
                        )
                    ),
                    sfcfd_rn[var_old],
                ],
                dim="time",
            )
            bcf_sel = xr_conf.sel(time=slice("%s-01-01" % (year), "%s-12-31" % (nyear)))
            # create new dataset
            bcf_sel = xr.Dataset({"%s" % (var_old): bcf_sel})
            # copy attributes
            bcf_sel = copyenv(bcf_sel, esmf, var_old, var_old)
            # change values
            esmf[var_old] = bcf_sel[var_old]

        elif y == 2010:
            xr_conf = xr.concat(
                [
                    sfcfd_rn[var_old],
                    esmf[var_old].sel(
                        time=slice("%s-01-01" % ("2013"), "%s-12-31" % ("2014"))
                    ),
                ],
                dim="time",
            )
            bcf_sel = xr_conf.sel(time=slice("%s-01-01" % (year), "%s-12-31" % (nyear)))
            # create new dataset
            bcf_sel = xr.Dataset({"%s" % (var_old): bcf_sel})
            # copy attributes
            bcf_sel = copyenv(bcf_sel, esmf, var_old, var_old)
            # change values
            esmf[var_old] = bcf_sel[var_old]

        else:
            bcf_sel = sfcfd_rn.sel(
                time=slice("%s-01-01" % (year), "%s-12-31" % (nyear))
            )
            # copy attributes
            bcf_sel = copyenv(bcf_sel, esmf, var_old, var_old)
            # change values
            esmf[var_old] = bcf_sel[var_old]

        # None encoding
        esmf.time_bnds.encoding["_FillValue"] = None

    # return reformatted dataset
    return esmf


def reformat_and_save_3d(
    bc_path,
    tlevel,
    startyear,
    endyear,
    input_vargcm,
    origin_vargcm,
    out_path,
    infor,
    gname,
    period,
    cinfor,
    sinfor,
):
    """
    Reformat and save 3D bias-corrected data to netCDF files.

    Parameters:
    bc_path (str): Path to the bias-corrected data files.
    tlevel (int): Total number of vertical levels.
    startyear (int): The first year of the data.
    endyear (int): The last year of the data.
    input_vargcm (list): List of input variable names from the GCM.
    origin_vargcm (list): List of original variable names from the GCM.
    out_path (str): Path to save the output netCDF files.
    infor (str): Additional information for the output file names.
    gname (str): Name of the GCM model.
    period (str): Time period information for the output file names.
    cinfor (str): Additional information for the output file names.
    sinfor (str): Additional information for the output file names.

    Returns:
    None: Saves the reformatted data to netCDF files in the specified output path.
    """

    # Load appropriate time periods
    years = np.arange(startyear, endyear + 1, 1)  # whole period
    months = np.arange(1, 13, 1)  # months
    levels = np.arange(1, tlevel + 1, 1)  # total number of vertical levels

    # Load bias-corrected data
    list_3D = [
        sorted(glob.glob(os.path.join(bc_path, f"3d.bcd.{idx}.output.nc")))
        for idx in range(1, tlevel + 1)
    ]
    ifile_3D = ["".join(list_3D[i]) for i in range(tlevel)]
    # test
    # Open the bias-corrected data with xarray
    partial_func = partial(add_lev_dim)
    bcf = xr.open_mfdataset(
        ifile_3D,
        preprocess=partial_func,
        concat_dim="lev",
        chunks={"time": 1000},
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=True,
        combine="nested",
    )

    # update time dimension
    time = pd.date_range(f"{startyear}-01-01", freq="6H", periods=len(bcf.time))
    bcf = bcf.update({"time": time})

    # Loop through the input variables and years to reformat and save the data
    # Multiprocessing has not been used due to memory issue.
    for k in range(0, len(input_vargcm) - 1):
        for y in range(startyear - 1, endyear + 1):
            year = str(y)
            nyear = y + 1
            nyear = str(nyear)
            esmf = reformatsave_3d(bcf, input_vargcm[k], origin_vargcm[k], y)

            # save data
            print("Save 3d to netcdf ", y)
            esmf.load().to_netcdf(
                f"{out_path}{origin_vargcm[k]}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{year}01010600-{nyear}01010000.nc"
            )
            print("Completed ", y)


def reformat_and_save_2d(
    bc_path,
    startyear,
    endyear,
    input_vargcm,
    origin_vargcm,
    out_path,
    infor,
    gname,
    period,
    cinfor,
    sinfor,
    starty_sfc,
    endy_sfc,
):
    """
    Reformat and save 2D bias-corrected data to netCDF files.

    Parameters:
    bc_path (str): Path to the bias-corrected data files.
    startyear (int): The first year of the data.
    endyear (int): The last year of the data.
    input_vargcm (list): List of input variable names from the GCM.
    origin_vargcm (list): List of original variable names from the GCM.
    out_path (str): Path to save the output netCDF files.
    infor (str): Additional information for the output file names.
    gname (str): Name of the GCM model.
    period (str): Time period information for the output file names.
    cinfor (str): Additional information for the output file names.
    sinfor (str): Additional information for the output file names.
    starty_sfc (int): The first year indicated in the name of the original surface data file.
    endy_sfc (int): The last year indicated in  the name of of the original surface data file.

    Returns:
    None: Saves the reformatted data to netCDF files in the specified output path.
    """

    # ================ Start creating input files, Surface fields ======================#
    with xr.open_dataset(f"{bc_path}sfc.bcd.output.nc") as sfcf:
        # update time dimension
        time = pd.date_range(
            "%s-01-01" % (startyear), freq="6H", periods=len(sfcf.time)
        )
        sfcf = sfcf.update({"time": time})
        # resample to daily
        sfcfd = sfcf.resample(time="D").mean("time")

    # Loop through the 2d input variable and years to reformat and save the data
    # Multiprocessing has not been used due to memory issue.
    for y in range(starty_sfc, endy_sfc + 1, 10):
        esmf = reformatsave_2d(sfcfd, input_vargcm[-1], origin_vargcm[-1], y)
        year = str(y)
        nyear = y + 9
        if nyear > 2009:
            nyear = y + 4
        nyear = str(nyear)

        # save data
        print("Save 2d to netcdf ", y)
        esmf.load().to_netcdf(
            f"{out_path}{origin_vargcm[-1]}_Oday_{gname}_{period}_{cinfor}_{sinfor}_{year}0101-{nyear}1231.nc"
        )
        print("Completed ", y)

    # Load appropriate time periods


# ---------------------------------------------------------------------------------------------------
