<a name="toc"></a>
# <p align="center">SDMBC</p>

SDMBC is an open-source Python pacakge for correcting the full atmospheric fields as well as sea surface temperature of global climate model (GCM) datasets prior to dynamical downscaling.

<br/>

## Table of contents

1. [Acknowledgements](#acknowledgements)
1. [Background to SDMBC package](#background)
1. [Installation](#installing)
1. [Input data](#inputdata)
1. [Output data](#outputdata)
1. [Tutorial](#tutorial)
1. [Get in touch](#getintouch)

<br/>

<a name="acknowledgements"></a>
## 1. Acknowledgements
[RETURN TO TOP](#toc)

* SDMBC stands for the Sub-Daily Multivariate Bias Correction and was developed by [Youngil Kim](https://github.com/young-ccrc/). Theoretical support was provided by [Ashish Sharma](https://www.unsw.edu.au/staff/ashish-sharma) and [Jason Evans](https://www.unsw.edu.au/staff/jason-evans).

* The design of SDMBC was influenced by [Multivariate Bias Correction (MBC) R pakcage](https://www.sciencedirect.com/science/article/pii/S1364815217309684#ec-research-data), which adjusts for the biases in inter-variable relationships of climate model outputs across multiple time scales.

* We acknowledge funding from the UNSW Scientia PhD Scholarship Scheme, the ARC Centre of Excellence for Climate Extremes (CE170100023), and the Australian Government under the National Environmental Science Program. This research was undertaken with the assistance of resources from the National Computational Infrastructure (NCI Australia), an NCRIS enabled capability supported by the Australian Government.

<br/>

<a name="background"></a>
## 1. Background to SDMBC package
[RETURN TO TOP](#toc)

Various bias correction techniques have been used to correct RCM input boundary conditions, ranging from simple scaling to more complex approaches that mimic observed multi-scale relationships in simulations. Studies have shown that increasingly sophisticated techniques that correct auto- and cross-dependence attributes can significantly improve rainfall characteristics, particularly for variability and extremes. Multivariate bias correction of RCM input boundary conditions has also been demonstrated to better represent compound events, which can be defined as a combination of extremes and hazards. However, while correcting the physical relationships between variables is important to reduce errors in simulated extreme events, daily and longer-term bias correction may impact the simulation of sub-daily rainfall, assuming that diurnal patterns are adequately modeled by the global climate model (GCM). Therefore, a novel approach that corrects multivariate relationships between variables at a sub-daily time scale was developed. RCM simulations with sub-daily multivariate bias-corrected boundary conditions demonstrated improved performance particularly for extreme events. However, the bias correction process requires complex mathematical formulations and large matrices, making it challenging for users to apply the method. None of studies have developed software to support this process.
To address this issue, we developed the first software package for sub-daily multivariate bias correction, called Sub-Daily Multivariate Bias Correction (SDMBC) for Dynamical Downscaling, in the Python environment. The approach has been proposed by (Kim 2023, under review), and is generally more effective at representing the range of diurnal rainfall magnitude compared to multivariate bias correction and corrects the full atmospheric fields and sea surface temperature of GCM datasets prior to dynamical downscaling. This software package simplifies the implementation of the bias correction process.

<br/>

<a name="installing"></a>
## 1. Installation
[RETURN TO TOP](#toc)

The following sections describe the software requirements and the process of installing SDMBC.

### 3.1 Software requirements (prerequisite):
* [Python](https://www.python.org/downloads/) version 3 or later.
* To generate output stastitics figures
   * the [*basemap*](https://matplotlib.org/basemap/users/installing.html)
   * the [*cartopy*](https://scitools.org.uk/cartopy/docs/latest/installing.html)

<br/>

### 3.2 Download
There are several options to donwload this package to your computer.

1. Download and extract [this file](https://github.com/young-ccrc/sdmbc/archive/refs/heads/master.zip) to your computer.
   This will create a directory named "sdmbc".

2. Install SDMBC directly from GitHub
```
pip install https://github.com/young-ccrc/sdmbc/archive/refs/heads/master.zip
```

3. Install SDMBC from PYPI
```
pip install sdmbc
```

4. Clone SDMBC git repository and install
```
git clone https://github.com/young-ccrc/sdmbc.git
```
Then,
```
python setup.py install
```

* Recommend to install this package under new python or conda environment.
Create and activate new conda environment
```
conda create -n sdmbc python=3.10
conda activate sdmbc
```

<br/>

<a name="inputdata"></a>
## 1. Input data
[RETURN TO TOP](#toc)

Users need to modify the ‘user_input.py’ file, which includes the input data information and bias correction options. The input files consist of global climate model (GCM) data to be bias-corrected and reanalysis data used here as an 'observation'. It should be noted that the reanalysis data should be properly interpolated to match the GCM resolutions prior to being used as inputs. To conduct bias correction, a minimum of 31 years simulation periods are recommended to correct climatological statistics in consideration of a 1-year spin-up period for RCM simulations.

Following files must be in the same folder.

* user_input.py
* main_biascorrection.exe
* input files
   * 3d.gcm.%level%.input.nc: three-dimensional atmospheric variables of GCM, including specific humidity (q), temperature (t), and zonal and meridional wind components (u and v)
   * 3d.obs.%level%.input.nc: three-dimensional atmospheric variables of observation, including specific humidity (q), temperature (t), zonal and meridional wind components (u and v)
   * sfc.gcm.input.nc: two-dimensional surface variable of GCM including sea surface temperatrue (sst)
   * sfc.obs.input.nc: two-dimensional surface variable of observation including sea surface temperatrue (sst)


<br/>

<a name="outputdata"></a>
## 1. Output data
[RETURN TO TOP](#toc)

This package provides two types of bias-corrected outputs: three-dimensional atmospheric variables at each vertical level and sea surface temperature. If bias correction performed properly, key statistics of the outputs within the selected domain will be generated. The results provide a model performance evaluation at different time scales: day, month, season, and year. The statistics cover climatological mean, standard deviation, lag1 auto-correlation, cross-correlation, mean absolute bias of sea surface temperature, and distribution at a sub-daily time scale.
* output files
   * 3d.bcd.%level%.output.nc: bias-corrected specific humidity (q), temperature (t), and zonal and meridional wind components (u and v)
   * sfc.bcd.output.nc: bias-corrected sea surface temperatrue (sst)

<br/>

<a name="tutorial"></a>
## 1. Totorial
[RETURN TO TOP](#toc)

Jupyter Notebook example, how to run this packge, can be found in example folder.

<br/>

<a name="getintouch"></a>
## 1. Get in touch
[RETURN TO TOP](#toc)

The authors welcome any contributions to code development going forward. youngil.kim@student.unsw.edu.au

<br/>
