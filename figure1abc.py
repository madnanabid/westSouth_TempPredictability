# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 21:44:01 2024

@author: Dell
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import cartopy.feature as cf
import cartopy.mpl.ticker as cticker
from matplotlib.patches import Rectangle
from scipy.stats import linregress
import geopandas as gpd
# Load topographic data from the netCDF file
dataset = Dataset('Data Path')
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]
topo_data = dataset.variables['btdata'][:]
topo_data = topo_data / 1000

# Specify the region of interest
lat_min, lat_max = 20, 41
lon_min, lon_max = 55, 91

# Find the indices corresponding to the specified region
lat_indices = np.where((lat >= lat_min) & (lat <= lat_max + 1.5))[0]
lon_indices = np.where((lon >= lon_min) & (lon <= lon_max + 1.5))[0]

# Subset the data to the specified region
subset_topo_data = topo_data[lat_indices[0]:lat_indices[-1] + 1, lon_indices[0]:lon_indices[-1] + 1]

# Plot settings
fig = plt.figure(figsize=(14, 10))

# Plot topography at the top
#ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax1.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
levels = [0, 0.5, 1, 2, 3, 4, 5, 6]
topo = ax1.contourf(lon[lon_indices], lat[lat_indices][::-1], subset_topo_data[::-1, :], levels=levels, cmap='turbo', extend='both')
ax1.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax1.coastlines()
ax1.add_feature(cf.BORDERS)
ax1.set_xticks(np.arange(55, 91, 5), crs=ccrs.PlateCarree())
ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax1.set_yticks(np.arange(20, 41, 5), crs=ccrs.PlateCarree())
ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())
ax1.add_patch(Rectangle((65, 31), 15, 8, fc='none', ec='black', lw=2))

# Add annotations
FS = fontsize = 7
plt.text(69, 29, 'Pakistan', fontsize=FS, color='black', weight='bold', rotation=45)
plt.text(57, 31, 'Iran', fontsize=FS, color='black', weight='bold')
plt.text(65, 32, 'Afghanistan', fontsize=FS, color='black', weight='bold', rotation=45)
plt.text(80, 35, 'China', fontsize=FS, color='black', weight='bold')
plt.text(76, 27, 'India', fontsize=FS, color='black', weight='bold')
plt.text(68.8, 38.5, 'Tajikistan', fontsize=FS, color='black', weight='bold')
plt.text(63.8, 40, 'Uzbekistan', fontsize=FS, color='black', weight='bold')
plt.text(57, 38.5, 'Turkmenistan', fontsize=FS, color='black', weight='bold')
plt.text(82, 28, 'Nepal', fontsize=FS, color='black', weight='bold')

# Add color bar for topography
cbar = plt.colorbar(topo, orientation='horizontal', shrink=0.4, pad=0.15)
cbar.ax.tick_params()
cbar.ax.set_title('Topography (KM)', loc="right")
plt.title("a) Regional Topographic Map", loc='left', fontsize=14)
kashmir=gpd.read_file('E:/DATA/topo/kashmir.shp')
ax1.add_geometries(kashmir['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.8)
# Annual Cycle: Temperature and Precipitation
# Prepare data
dir1 = "E:/DATA/ERA-DATA/ts/"
dir2 = "E:/DATA/ERA-DATA/prcp/"
tmpmn = nc.Dataset(dir1 + "ac.era5_tas.mon.1981-2022.1deg.nc", "r")
tmpstd = nc.Dataset(dir1 + "std.ac.era5_tas.mon.1981-2022.1deg.nc", "r")
prcpmn = nc.Dataset(dir2 + "ac.era5.prcp.1981_2022_new.nc", "r")
timevar = tmpmn.variables['time']
lonvar = tmpmn.variables['lon'][:]
latvar = tmpmn.variables['lat'][:]
tas = tmpmn.variables['t2m'][:,:,:]
tas = tas - 273.15
std = tmpstd.variables['t2m'][:,:,:]
prcp = prcpmn.variables['tp'][:,:,:]
prcp = prcp * 1000 * 30
nt, nlat, nlon = tas.shape
# Define region of interest
lonE = 65
lonW = 80
latS = 31
latN = 39
lon1 = np.where(lonvar == lonE)
lon2 = np.where(lonvar == lonW)
lat1 = np.where(latvar == latS)
lat2 = np.where(latvar == latN)

# Calculate weighted averages
lonx, latx = np.meshgrid(lonvar, latvar)
weights = np.cos(latx * np.pi / 180.)
tas_avg = np.zeros(nt)
prcp_avg = np.zeros(nt)
std_avg = np.zeros(nt)
for it in np.arange(nt):
    tas_avg[it] = np.ma.average(tas[it, int(lat1[0]):int(lat2[0]), int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]), int(lon1[0]):int(lon2[0])])
    prcp_avg[it] = np.ma.average(prcp[it, int(lat1[0]):int(lat2[0]), int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]), int(lon1[0]):int(lon2[0])])
    std_avg[it] = np.ma.average(std[it, int(lat1[0]):int(lat2[0]), int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]), int(lon1[0]):int(lon2[0])])

# Add subplot for the annual cycle
#ax2 = fig.add_subplot(2, 3, 2)
ax2 = fig.add_subplot(2, 2, 3)
# Plotting temperature data
x = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax2.plot(x, tas_avg, 'k-', marker='o')
ax2.set_ylabel("Temperature (°C)", fontsize=16)
ax2.set_ylim(-5, 25)
# Add secondary y-axis for precipitation
ax3 = ax2.twinx()
ax3.plot(x, prcp_avg, 'g-', marker='o')
ax3.set_ylabel("Precipitation (mm/month)", color='g', fontsize=14)
# Fill between lines
xmin = ['May', 'Jun']
ymin = [-5]
ymax = [25]
ax2.fill_between(xmin, ymin, ymax, alpha=0.3, color='tab:grey')
# Title and axis labels
plt.title("b) Annual Cycle Prcp and SATI (WSA)", loc='left', fontsize=14)
ax2.set_xlabel("Month", fontsize=14)

# Correlation Coefficient: SATI vs NINO34
# Prepare data
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
r_values = []
for month in months:
    dir1 = "E:/DATA/ERA-DATA/ts/mon/" + month + "/"
    dir2 = "E:/DATA/ERA-DATA/sst/mon/" + month + "/"
    tmp = nc.Dataset(dir1 + "dt.era5_tas." + month + ".1981-2022_1deg.nc", "r")
    sst = nc.Dataset(dir2 + "dt.era5_sst." + month + ".1981_2022.nc", "r")
    timevar = tmp.variables['time']
    lonvar = tmp.variables['lon'][:]
    latvar = tmp.variables['lat'][:]
    tas = tmp.variables['t2m'][:,:,:]
    nt, nlat, nlon = tas.shape
    mtmp = np.mean(tas, axis=0)
    anom = tas - mtmp
    lonE = 65
    lonW = 80
    latS = 31
    latN = 39
    lon3 = np.where(lonvar == lonE)
    lon4 = np.where(lonvar == lonW)
    lat3 = np.where(latvar == latS)
    lat4 = np.where(latvar == latN)
    lonx, latx = np.meshgrid(lonvar, latvar)
    weights = np.cos(latx * np.pi / 180.)
    tas_sati = np.zeros(nt)
    for it in np.arange(nt):
        tas_sati[it] = np.ma.average(
            anom[it, int(lat3[0]):int(lat4[0]), int(lon3[0]):int(lon4[0])],
            weights=weights[int(lat3[0]):int(lat4[0]), int(lon3[0]):int(lon4[0])])
    timevar1 = sst.variables['time']
    lonvar1 = sst.variables['lon'][:]
    latvar1 = sst.variables['lat'][:]
    tsst = sst.variables['sst'][:,:,:]
    msst = np.mean(tsst, axis=0)
    anomsst = tsst - msst
    lonES = 190
    lonWS = 240
    latSS = -5
    latNS = 5
    lon5 = np.where(lonvar1 == lonES)
    lon6 = np.where(lonvar1 == lonWS)
    lat5 = np.where(latvar1 == latSS)
    lat6 = np.where(latvar1 == latNS)
    lonx, latx = np.meshgrid(lonvar1, latvar1)
    weights = np.cos(latx * np.pi / 180.)
    tas_nino34_obs = np.zeros(nt)
    for it in np.arange(nt):
        tas_nino34_obs[it] = np.ma.average(
            anomsst[it, int(lat5[0]):int(lat6[0]), int(lon5[0]):int(lon6[0])],
            weights=weights[int(lat5[0]):int(lat6[0]), int(lon5[0]):int(lon6[0])])
    slope, intercept, r_value, p_value, std_err = linregress(tas_nino34_obs, tas_sati)
    r_values.append(r_value)

# Create the third subplot
#ax3 = fig.add_subplot(2, 3, 5)
ax3 = fig.add_subplot(2, 2, 4)
ax3.hlines(0, 0, 11, color='black', linewidth=0.5)
ax3.hlines(0.29, 0.29, 11, color='grey', linestyle='--', linewidth=0.5)
ax3.hlines(-0.29, -0.29, 11, color='grey', linestyle='--', linewidth=0.5)
ax3.plot(months, r_values, color='red', marker='o', linewidth=1)
xmin = [4, 5]
ymin = [-0.5]
ymax = [0.5]
ax3.fill_between(xmin, ymin, ymax, alpha=0.3, color='tab:grey')
ax3.set_xlabel('Month', fontsize=14)
ax3.set_ylim(-0.4, 0.4, 0.2)
ax3.set_ylabel('Correlation Coefficient', fontsize=14)
ax3.set_title("c) WSA(SATI) vs NINO34", loc='left', fontsize=14)
ax3.set_title("ERA5", loc='right', fontsize=14)

plt.tight_layout()
plt.savefig('figure1abc.png', dpi=300)
plt.show()
