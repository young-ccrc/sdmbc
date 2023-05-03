#! usr/bin/python
# =======================================================================================
# Description
#
# Input file formats
# - Atmospheric fields: 3D.%model%.%idx%.input.nc
# - Surface fields: sfc.%model%.input.nc
# Output file formats
# - Atmospheric fields: %variable%_6hrLev_ACCESS-ESM1-5_historical_r1i1p1f1_gn_1982-2012.nc
# - Surface fields: tos_6hrLev_ACCESS-ESM1-5_historical_r1i1p1f1_gn_1982-2012.nc

# This script, named main_biascorrection.py, performs bias correction
# of GCM outputs.
# The script first reads input parameters from a file named biascorrection_input.dat.
# If the biascorrection parameter is set to 1, the script reads necessary inputs,
# modifies the input file with the new values, and runs an executable file named
# main_biascorrection.exe using subprocess module.
# The script then checks for two other parameters, draw_figure and reformat,
# and runs the corresponding Python scripts named main_figure.py and main_reformat.py,
# respectively, if their respective parameters are set to 1.
# Finally, the script outputs a start and end message indicating the start and end
# of the script execution.

# Requirements
# This script requires the following libraries:
# xarray
# subprocess

# Written by Youngil(Young) Kim
# PhD Candidate
# Water Research Centre
# Climate Change Research Centre
# University of New South Wales
# 2023-03-20
#
# =======================================================================================
import pytest
import xarray as xr
from user_input import *
import subprocess
from analysis_plot import AnalysisBC
from output_reformatter import reformat_and_save_3d, reformat_and_save_2d


class BiasCorrection:
    """
    A class to handle bias correction of climate variables.

    Attributes:
    bc_path (str): Path to the bias-corrected data.
    level (int): Vertical level of the data.
    startyear (int): Start year of the data.
    endyear (int): End year of the data.
    no_of_variables (int): Number of variables.
    time_scale (int): Time scale of the data.
    no_of_iterations (int): Number of iterations for the bias correction.
    missing_value (float): Missing value in the dataset.
    lon_range (tuple): Tuple of the form (min_lon, max_lon) for longitude range.
    lat_range (tuple): Tuple of the form (min_lat, max_lat) for latitude range.
    moving_window (int): Moving window for the bias correction.
    correction_model (int): Correction model to be used for bias correction.
    sub_daily_correction (int): Sub-daily correction to be applied.
    numberofcores (int): Number of cores to be used for parallel processing.
    lower_limit (list): List of lower limits for variables.
    upper_limit (list): List of upper limits for variables.
    """

    def __init__(
        self,
        bc_path,
        level,
        startyear,
        endyear,
        no_of_variables,
        time_scale,
        no_of_iterations,
        missing_value,
        lon_range,
        lat_range,
        moving_window,
        correction_model,
        sub_daily_correction,
        lower_limit,
        upper_limit,
    ):
        self.bc_path = bc_path
        self.level = level
        self.startyear = startyear
        self.endyear = endyear
        self.no_of_variables = no_of_variables
        self.time_scale = time_scale
        self.no_of_iterations = no_of_iterations
        self.missing_value = missing_value
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.moving_window = moving_window
        self.correction_model = correction_model
        self.sub_daily_correction = sub_daily_correction
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def perform_bias_correction(self, correction, figure, reformat):
        """
        Perform the bias correction using the specified parameters.

        Reads the "biascorrection_input.dat" file and modifies the input parameters
        based on the instance attributes.
        Then runs the "main_biascorrection.exe" program with the updated input file.

        Attributes:
        -----------
        correction : optional
        If True (1), statistics of the outputs will be calculated, and plots will be saved.
        otherwise not. Default is False (0).

        figure : optional
        If True (1), statistics of the outputs will be calculated, and plots will be saved.
        otherwise not. Default is False (0).

        kstest : bool, optional
        If True (1), the Kolmogorov-Smirnov (K-S) test will be performed,
        otherwise not. Default is False (0).

        reformat : optional
        If True (1), the outputs will be reformatted as the original forms.
        otherwise not. Default is False (0).
        -----------
        """

        # Open biascorrection input file
        with open(f"{self.bc_path}biascorrection_input.dat", "r") as file:
            newlines = file.readlines()

        # Load gcm
        gcm_input = xr.open_dataset(
            self.bc_path + "3d.gcm." + str(self.level) + "." + "input" + ".nc"
        )

        # New values to replace the old ones
        self.no_of_years = self.endyear - self.startyear + 1

        newlines[2] = "    \t\t {:2d}                   {}\n".format(
            self.no_of_years, self.startyear
        )
        newlines[5] = "    \t\t {:2d}                   {}\n".format(
            self.no_of_years, self.startyear
        )
        newlines[7] = "\t{}\n".format(self.no_of_variables)
        newlines[9] = "\t{}\n".format(self.time_scale)
        newlines[11] = "\t{}\n".format(self.no_of_iterations)
        newlines[13] = "\t{}\n".format(self.missing_value)
        newlines[15] = "\t{}\t{}\t{}\n".format(
            len(gcm_input.lon), len(gcm_input.lat), len(gcm_input.time)
        )
        newlines[17] = "\t{}\t{}\t{}\t{}\n".format(
            self.lon_range[0], self.lon_range[1], self.lat_range[0], self.lat_range[1]
        )
        newlines[19] = "\t{}\n".format(self.level)
        newlines[21] = "\t{}\n".format(self.moving_window)
        newlines[25] = "\t{}\n".format(self.correction_model)
        newlines[27] = "\t{}\n".format(self.sub_daily_correction)
        #        newlines[29] = "\t{}\n".format(self.numberofcores)

        newlines[30] = "   \t\t1            {}     {}\t\t\n".format(
            self.lower_limit[0], self.upper_limit[0]
        )
        newlines[31] = "   \t\t1            {}     {}\t\t\n".format(
            self.lower_limit[1], self.upper_limit[1]
        )
        newlines[32] = "   \t\t1            {}     {}\t\t\n".format(
            self.lower_limit[2], self.upper_limit[2]
        )
        newlines[33] = "   \t\t1            {}     {}\t\t\n".format(
            self.lower_limit[3], self.upper_limit[3]
        )

        # Open the file in write mode and write the modified contents
        with open(f"{self.bc_path}biascorrection_input.dat", "w") as file:
            file.writelines(newlines)
        print("Update biascorrection_input.dat file")

        self.correction = correction
        self.figure = figure
        # self.kstest = kstest
        self.reformat = reformat

        # print('Start bias correction')
        if self.correction == 1:
            print("Start bias correction")
            # Replace 'program.exe' with the name of your .exe file
            program_name = f"{self.bc_path}main_biascorrection.exe"
            # Use subprocess to run the .exe file
            subprocess.run(program_name)

        if self.figure == 1:
            print("Drawing figures")
            if self.sub_daily_correction == 1:
                print("K-S test has been included")
                statistics = AnalysisBC(
                    bc_path,
                    lat_range,
                    lon_range,
                    startyear,
                    out_figure_path,
                    kstest=True,
                )
            if self.sub_daily_correction == 0:
                print("K-S test has not been included")
                statistics = AnalysisBC(
                    bc_path,
                    lat_range,
                    lon_range,
                    startyear,
                    out_figure_path,
                    kstest=False,
                )
            statistics.figure_atmos()
            print("Finish 3d field")
            if (self.level == 1) and (self.no_of_variables >= 4):
                statistics.figure_surface()
                print("Finish 2d field")

        if self.reformat == 1:
            print("Start reformatting")

            reformat_and_save_3d(
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
            )
            print("Finish 3D reformatting")

            if (self.level == 1) and (self.no_of_variables >= 4):
                reformat_and_save_2d(
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
                )
                print("Finish 2D reformatting")
