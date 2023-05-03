# ============================= Input =============================
# File path -------------------------------------------------
# Should start and end with "/"
bc_path = '/srv/ccrc/RegClim/z5239661/data53/sdmbc/example/'         # input file path where bias-corrected file exists.
out_path = '/srv/ccrc/RegClim/z5239661/data53/sdmbc/example/'        # output file path. 
out_figure_path = '/srv/ccrc/RegClim/z5239661/data53/sdmbc/example/' # output figure file path.
gcm_path = '/srv/ccrc/RegClim/z5239661/data53/esm1.5/input/'           # raw GCM file path. Required only for reformating process.

# Input variable names --------------------------------------------
origin_vargcm = ['hus','ta','ua','va','tos']    # original gcm variable name list
input_vargcm = ['q','t','u','v','sst']          # default, input variable's name for bias correction

# GCM informatin --------------------------------------------------
# CMIP6 global attritube description can be found in  
# https://docs.google.com/document/d/1h0r8RZr_f3-8egBMMh7aqLwy3snpD6_MrDz1q8n5XUk/edit
infor = '6hrLev'           # time scale 
gname = 'ACCESS-ESM1-5'    # gcm 
period = 'historical'      # experiment identifier
cinfor = 'r1i1p1f1'        # parent variant label
sinfor = 'gn'              # grid identifier
# ex: gcm = '_6hrLev_ACCESS-ESM1-5_historical_r1i1p1f1_gn_'

# Simulation periods ----------------------------------------------
startyear = 1982          # start year
endyear = 2012            # end year
starty_sfc = 1980         # start year, SST. Required only for reformating, and denpends on sst input file name. 
                          # for example, tos (sst) files of ACCESS-ESM1.5 are stored every 10 years (1980 - 1989) on NCI, 
                          # meaning that to reformat sst to original (tos) format, start year of sst should be 1980, not 1982.
endy_sfc = 2012           # end year, SST. Required only for reformating. Depends on GCMs. 
level = 1                 # vertical level to be corrected. 
tlevel = 38               # total number of vertal levels of gcm used. This will be used for loop.

# Bias correction information -------------------------------------

no_of_variables = 4       # number of input variable (at surface: w, t, q, sst, else: w, t, q)
time_scale = 0            # input data time scale, 0: daily, 1: monthly
no_of_iterations = 3      # number of bias correction iteration, three are recommended
missing_value = -999.0    # missing value identifier

# Selected area
lon_range = [1,6]      # should be integer, indicating grid cells (x-axis), not coordinates
lat_range = [1,9]       # should be integer, indicating grid cells (y-axis), not coordinates 

moving_window = 15              # centred moving window if input data is at daily time scale
correction_model = 4            # 1: mean, 2: mean and variance, 3: nested, 4: multivariate
sub_daily_correction = 1        # sub-daily bias correction, Yes: 1, No: 0

# Physical upper limit (wind speed (m/s), temperature (K), specific humidity (g/kg) , sea surface temperature (K))
upper_limit = [190,330,30,330]       
lower_limit = [0.001,137,0.001,270]

# Options for bias correction
correction = 1             # compute bias correction, Yes: 1, No: 0
draw_figure = 1            # draw sample statistics of the outputs, Yes: 1, No: 0
reformat_to_original = 0   # reformat the output files to original gcm form, Yes: 1, No: 0
                           # this should be 0 until the bias correction is done for all of the vertical levels, as the vertical levels should be the same as those of original GCM file.

#======================== User inputs end =========================