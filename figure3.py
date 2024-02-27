# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 21:44:01 2024

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib.patches import Rectangle
import netCDF4 as nc
from cartopy.util import add_cyclic_point
import cartopy.mpl.ticker as cticker
from scipy.stats import linregress
from scipy import stats
# Load ERA5 data
path_era5 = 'Data Path'
filename_era5 = 'filename'
ds_era5 = xr.open_dataset(path_era5 + filename_era5)
lat_era5 = ds_era5['lat'][:]
lon_era5 = ds_era5['lon'][:]
data_era5 = ds_era5['ev'][0, :, :]

# Load SEAS5 data
path_seas5 = 'filename'
filename_seas5 = 'filename'
ds_seas5 = xr.open_dataset(path_seas5 + filename_seas5)
lat_seas5 = ds_seas5['lat'][:]
lon_seas5 = ds_seas5['lon'][:]
data_seas5 = ds_seas5['ev'][0, :, :] * -1

# Plot settings
fig = plt.figure(figsize=(14, 12))

# Plot topography at the top
ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())

# Plot ERA5 data
ax1.set_extent((60, 85.2, 23, 40), crs=ccrs.PlateCarree())
data_era5, lons_era5 = add_cyclic_point(data_era5, coord=lon_era5)
ax1.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax1.coastlines()
ax1.add_feature(cf.BORDERS)
cs1 = ax1.contourf(lons_era5, lat_era5, data_era5, np.arange(-1.4, 1.5, 0.2),
                      transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r, extend='both')
ax1.add_patch(Rectangle((65, 31), 15, 8, fc='none', ec='black', lw=5))


# Define the xticks for longitude
ax1.set_xticks(np.arange(60,85.2,5), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
plt.xticks(fontsize=18) 
# Define the yticks for latitude
ax1.set_yticks(np.arange(23,40,3), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax1.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=18) 
ax1.set_title("a) EOF1: Var = 45%", loc='left', fontsize=14)
ax1.set_title("ERA5", loc='right', fontsize=14)

# Plot SEAS5 data
ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
ax2.set_extent((60, 85.2, 23, 40), crs=ccrs.PlateCarree())
data_seas5, lons_seas5 = add_cyclic_point(data_seas5, coord=lon_seas5)
ax2.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax2.coastlines()
ax2.add_feature(cf.BORDERS)
cs2 = ax2.contourf(lons_seas5, lat_seas5, data_seas5, np.arange(-1.4, 1.5, 0.2),
                      transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r, extend='both')
ax2.add_patch(Rectangle((65, 31), 15, 8, fc='none', ec='black', lw=5))

# Define the xticks for longitude
ax2.set_xticks(np.arange(60,85.2,5), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)
plt.xticks(fontsize=18) 
# Define the yticks for latitude
ax2.set_yticks(np.arange(23,40,3), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax2.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=18) 
ax2.set_title("b) EOF1: Var = 38%", loc='left', fontsize=14)
ax2.set_title("SEAS5", loc='right', fontsize=14)

# Create a colorbar for the first two subplots
cbar_ax = fig.add_axes([0.2, 0.5, 0.6, 0.04])  # [left, bottom, width, height]
#cbar_ax = fig.add_axes([0.1, 0.5, 0.8, 0.03])
cbar = fig.colorbar(cs1, cax=cbar_ax, orientation='horizontal')
cbar.ax.set_title('Temperature(°C)', fontsize=14)
cbar.ax.tick_params(labelsize=14)

#------------------------Timeseries------------------
# Plot SATI data
dir1 = "Data Path"
dir2 = "Data Path"
dir3 = "Data Path"

pc1_obs = nc.Dataset(dir1 + "filename", "r")
pc1_mod = nc.Dataset(dir2 + "filename", "r")
tmp = nc.Dataset(dir3 + "filename", "r")

time1 = pc1_obs.variables['time'][:]
time2 = pc1_mod.variables['time'][:]
tas_obs = pc1_obs.variables['pc1'][0:42, :, :]
tas_mod = pc1_mod.variables['pc1'][:, :, :]
mpc1 = np.average(tas_mod.reshape(-1, 42), axis=0)
opc1 = tas_obs[:, 0, 0]
timevar = tmp.variables['time']
lonvar = tmp.variables['lon'][:]
latvar = tmp.variables['lat'][:]
tas = tmp.variables['t2m'][:,:,:]

msst=np.mean(tas,axis=0)
#print(msst)
#print(msst.shape)
anom=[]
anom=tas-msst
#print(msst_obs.shape) 
nt, nlat, nlon = tas.shape
ngrd = nlon * nlat
lonE = 65
lonW = 80
latS = 31
latN = 39
lon1 = np.where(lonvar == lonE)
lon2 = np.where(lonvar == lonW)
lat1 = np.where(latvar == latS)
lat2 = np.where(latvar == latN)
lonx, latx = np.meshgrid(lonvar, latvar)
weights = np.cos(latx * np.pi / 180.)
tas_avg = np.zeros(nt)
for it in np.arange(nt):
    tas_avg[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]), int(lon1[0]):int(lon2[0])],
                                 weights=weights[int(lat1[0]):int(lat2[0]), int(lon1[0]):int(lon2[0])])
print(tas_avg.shape) 
# mpc1_sati = np.average(tas_avg.reshape(-1, 42), axis=0)
#ax2 = fig.add_subplot(2, 3, 2)
import matplotlib.gridspec as gridspec
# Define the height ratios for the subplots
height_ratios = [2, 1.5]
# Create a gridspec to control the subplot layout
gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios)
# Plot SATI data
ax3 = fig.add_subplot(gs[1], aspect=4)
ax3.plot(np.arange(1981, 2023, 1), opc1, color='brown', label='PC1(ERA5)', linewidth=1)
ax3.plot(np.arange(1981, 2023, 1), mpc1 * -1, color='green', label='PC1(SEAS5)', linewidth=1)
ax3.plot(np.arange(1981, 2023, 1), tas_avg, color='black', label='SATI', linestyle='dashed', linewidth=1)
ax3.hlines(0, xmin=1981, xmax=2023, color='grey', linestyle='-', linewidth=0.5)

slope, intercept, r_value, p_value, std_err = linregress(opc1,mpc1*-1)
print('PCs=','Slope: %.2f' % slope, 'Intercept: %.2f' %intercept,'r value: %.2f' %r_value)

slope1, intercept1, r_value1, p_value1, std_err1 = linregress(opc1,tas_avg)
print('SATI=','Slope: %.2f' % slope1, 'Intercept: %.2f' %intercept1,'r value: %.2f' %r_value1)
#cor = tas_obs.corr(tas_mod)
label = "{:.2f}".format(r_value)
res = stats.pearsonr(opc1,mpc1*-1)
#label1 = "{:.2f}".format(res)

print(res)
# cor2 = data.PC1ERA.corr(sati)
# label2 = "{:.2f}".format(cor2)
# cor3 = data.NINO34obs.corr(data.SATI)
label2 = "{:.2f}".format(r_value1)
#print(label)
# print(label2)
# print(label3)
plt.text(1982,2.3, 'CC: PC1(ERA5) vs PC1(SEAS5) = (%s)'%(label), fontsize=12)
plt.text(1982,2, 'CC: PC1(ERA5) vs SATI      = (%s)'%(label2), fontsize=12)
ax3.set_ylim(-2.5, 2.5, 0.5)
ax3.set_ylabel('Temperature (°C)', fontsize=14)
ax3.set_xlim(1981, 2023, 3)
ax3.legend(loc=0, fontsize=12)
ax3.set_title("c) ERA5 vs ECMWF-SEAS5", loc='left', fontsize=14)
# Adjust xticks and yticks font size
ax3.tick_params(axis='x', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)

plt.tight_layout()
plt.savefig('figure3abcd.png', dpi = 300)
plt.show()
