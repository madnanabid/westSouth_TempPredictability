# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 23:30:29 2024

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import netCDF4 as nc

# File paths
era_data_path = 'E:/DATA/ERA-DATA/ts/MJ/'
era_file_name = 'era5_tas.MJ.1981-2022_1deg.nc'
seas_data_path = 'E:/DATA/ECMWF-SYS5/t2m/cat/'
seas_file_name = 'ensmean.sys5.t2m.MJ.1981-2022.nc'

# Open ERA data
ds_era = xr.open_dataset(era_data_path + era_file_name, decode_times=False)
ds_era = ds_era - 273.15  # Convert to Celsius
ds_era_mean = ds_era.mean(dim='time')
# Calculate standard deviation
ds_era_std = ds_era.std(dim='time')

# Open SEAS5 data
ds_seas = xr.open_dataset(seas_data_path + seas_file_name)
ds_seas = ds_seas - 273.15  # Convert to Celsius
ds_seas_mean = ds_seas.mean(dim='time')
# Calculate standard deviation
ds_seas_std = ds_seas.std(dim='time')

fig = plt.figure(figsize=(12, 10))

# Plot CLM ERA data
ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax1.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
data_era, lons_era = add_cyclic_point(ds_era_mean['t2m'], coord=ds_era['lon'])
ax1.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax1.coastlines()
ax1.add_feature(cf.BORDERS)
cmap_era = plt.cm.RdBu_r
cs1 = ax1.contourf(lons_era, ds_era['lat'], data_era, np.arange(-10, 41, 5), transform=ccrs.PlateCarree(), 
                   norm=colors.CenteredNorm(), cmap=cmap_era, extend='both')
ax1.set_xticks(np.arange(55, 91, 5), crs=ccrs.PlateCarree())
ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax1.set_yticks(np.arange(20, 41, 5), crs=ccrs.PlateCarree())
ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())
# Adjust the fontsize of xticks and yticks
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
# Annotations
# countries = {'Pakistan': (67.0, 30.0), 'Iran': (57.0, 31.0), 'Afghanistan': (61.0, 34.0), 
#              'China': (80.0, 35.0), 'India': (76.0, 27.0), 'Tajikistan': (67.8, 38.5), 
#              'Uzbekistan': (63.3, 40), 'Turkmenistan': (57, 38), 'Nepal': (81, 28)}
# for country, (x, y) in countries.items():
#     if country == 'Turkmenistan':
#         ax1.text(x, y, f' {country}', fontsize=8, color='black', weight='bold', rotation=-10)
#     elif country == 'Uzbekistan':
#         ax1.text(x, y, f' {country}', fontsize=8, color='black', weight='bold')
#     elif country == 'Tajikistan':
#         ax1.text(x, y, f' {country}', fontsize=8, color='black', weight='bold')    
#     else:
#         ax1.text(x, y, f' {country}', fontsize=10, color='black', weight='bold')

# Titles
ax1.set_title("a) SAT", loc='left', fontsize=14)
ax1.set_title("ERA5", loc='right', fontsize=14)

# # Colorbar for ERA data
# cbar_ax1 = fig.add_axes([0.12, 0.47, 0.35, 0.02])  # [left, bottom, width, height]
# cbar1 = fig.colorbar(cs1, cax=cbar_ax1, orientation='horizontal')
# cbar1.ax.tick_params(labelsize=14) 
# cbar1.ax.set_xlabel('Temperature (°C)', fontsize=14)

# Plot CLM SEAS5 data
ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
ax2.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
data_seas, lons_seas = add_cyclic_point(ds_seas_mean['t2m'], coord=ds_seas['lon'])
ax2.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax2.coastlines()
ax2.add_feature(cf.BORDERS)
cmap_seas = plt.cm.RdBu_r
cs2 = ax2.contourf(lons_seas, ds_seas['lat'], data_seas, np.arange(-10, 41, 5), transform=ccrs.PlateCarree(), 
                   norm=colors.CenteredNorm(), cmap=cmap_seas, extend='both')
ax2.set_xticks(np.arange(55, 91, 5), crs=ccrs.PlateCarree())
ax2.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax2.set_yticks(np.arange(20, 41, 5), crs=ccrs.PlateCarree())
ax2.yaxis.set_major_formatter(cticker.LatitudeFormatter())
# Adjust the fontsize of xticks and yticks
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
# Annotations
# for country, (x, y) in countries.items():
#     if country == 'Turkmenistan':
#         ax2.text(x, y, f' {country}', fontsize=8, color='black', weight='bold', rotation=-10)
#     elif country == 'Uzbekistan':
#         ax2.text(x, y, f' {country}', fontsize=8, color='black', weight='bold')
#     elif country == 'Tajikistan':
#         ax2.text(x, y, f' {country}', fontsize=8, color='black', weight='bold')  
#     else:
#         ax2.text(x, y, f' {country}', fontsize=10, color='black', weight='bold')
# Titles
ax2.set_title("b) SAT", loc='left', fontsize=14)
ax2.set_title("SEAS5", loc='right', fontsize=14)

# Colorbar
cbar_ax = fig.add_axes([0.2, 0.47, 0.6, 0.04])  # [left, bottom, width, height]
cbar = fig.colorbar(cs2, cax=cbar_ax, orientation='horizontal')
# cbar.ax.tick_params(labelsize=14) 
# cbar.ax.set_xlabel('Temperature (°C)', fontsize=14)
cbar.ax.tick_params(labelsize=12) 
cbar.ax.set_title('Mean Temperature (°C)',fontsize=14)

#---------------------STD---------------------

# File paths
era_data_path = 'E:/DATA/ERA-DATA/ts/MJ/'
era_file_name = 'dt.era5_tas.MJ.1981-2022_1deg.nc'
# seas_data_path = 'i:/DATA/ECMWF-SYS5/t2m/cat/dt/'
# seas_file_name = 'ensmean.sys5.t2m.MJ.1981-2022.nc'

# Open ERA data
ds_era = xr.open_dataset(era_data_path + era_file_name, decode_times=False)
ds_era = ds_era - 273.15  # Convert to Celsius
ds_era_mean = ds_era.mean(dim='time')
# Calculate standard deviation
ds_era_std = ds_era.std(dim='time')

# # Open SEAS5 data
# ds_seas = xr.open_dataset(seas_data_path + seas_file_name)
# ds_seas = ds_seas - 273.15  # Convert to Celsius
# ds_seas_mean = ds_seas.mean(dim='time')
# # Calculate standard deviation
# ds_seas_std = ds_seas.std(dim='time')

# Plot STD ERA data

ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
ax3.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
data_era_std, lons_era = add_cyclic_point(ds_era_std['t2m'], coord=ds_era['lon'])
ax3.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax3.coastlines()
ax3.add_feature(cf.BORDERS)
cmap_std_era = plt.cm.YlOrBr
cs3 = ax3.contourf(lons_era, ds_era['lat'], data_era_std, np.arange(0.2, 2, 0.2), transform=ccrs.PlateCarree(), 
                    cmap=cmap_std_era, extend='both')
ax3.set_xticks(np.arange(55, 91, 5), crs=ccrs.PlateCarree())
ax3.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax3.set_yticks(np.arange(20, 41, 5), crs=ccrs.PlateCarree())
ax3.yaxis.set_major_formatter(cticker.LatitudeFormatter())
# Adjust the fontsize of xticks and yticks
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=12)
# Titles
ax3.set_title("c) STD", loc='left', fontsize=14)
ax3.set_title("ERA5", loc='right', fontsize=14)

# Plot STD SEAS5 data
# Data path
diri = "E:/DATA/ECMWF-SYS5/t2m/cat/dt/"

# List of ensemble file names
file_names = ["sys5_e1.nc", "sys5_e2.nc", "sys5_e3.nc", "sys5_e4.nc", "sys5_e5.nc",
              "sys5_e6.nc", "sys5_e7.nc", "sys5_e8.nc", "sys5_e9.nc", "sys5_e10.nc",
              "sys5_e11.nc", "sys5_e12.nc", "sys5_e13.nc", "sys5_e14.nc", "sys5_e15.nc",
              "sys5_e16.nc", "sys5_e17.nc", "sys5_e18.nc", "sys5_e19.nc", "sys5_e20.nc",
              "sys5_e21.nc", "sys5_e22.nc", "sys5_e23.nc", "sys5_e24.nc", "sys5_e25.nc"]

# Load ensemble data into an array
ensemble_data = []
for file_name in file_names:
    file_path = diri + file_name
    data = nc.Dataset(file_path, "r")
    timevar = data.variables['time']
    lonvar = data.variables['lon'][:]
    latvar = data.variables['lat'][:]
    may = data.variables['m1'][:, :, :]
    jun = data.variables['m2'][:, :, :]
    mj = (may + jun) / 2
    ensemble_data.append(mj)  # Assuming "t2m" is the variable name for temperature

ensemble_data = np.array(ensemble_data)
print(ensemble_data.shape)

ensemble_mean = np.mean(ensemble_data, axis=0)
print(ensemble_mean.shape)

time_mean = np.mean(ensemble_mean, axis=0)
# print(time_mean.shape)
#ensemble_variances = np.mean(np.power((ensemble_data - time_mean), 2),axis = 0)
# print(diff_squared.shape)
# Calculate the variance of each ensemble member
ensemble_variances = np.zeros((25,181,360),np.float32)
for i in range(25):
    ensemble_variances[i,:,:] = np.var(ensemble_data[i,:,:,:], axis=0)
print("Shape of ensemble_variances:", ensemble_variances.shape)
# # Calculate the mean of ensemble variances
ens_variance = np.mean(ensemble_variances, axis = 0)
#print(ensemble_variance.shape)

std_mod = np.sqrt(ens_variance)
print("Value of sqrt_mean_ensemble_variance:", std_mod.shape)
#smoothed_data = uniform_filter(std_mod, size=3)

#ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())
# Set the axes using the specified map projection
#ax1=plt.axes(projection=ccrs.PlateCarree())
#ax1.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
# Add cyclic point to data
data1=std_mod
ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())
ax4.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
data1, lons = add_cyclic_point(data1, coord=data.variables['lon'][:])
#data1, lons_seas = add_cyclic_point(data1, coord=ds_seas['lon'])
ax4.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax4.coastlines()
ax4.add_feature(cf.BORDERS)
cmap_std_seas = plt.cm.YlOrBr
cs4 = ax4.contourf(lons, data.variables['lat'][:], data1, np.arange(0.2, 2, 0.2), transform=ccrs.PlateCarree(), 
                    cmap=cmap_std_seas, extend='both')
ax4.set_xticks(np.arange(55, 91, 5), crs=ccrs.PlateCarree())
ax4.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax4.set_yticks(np.arange(20, 41, 5), crs=ccrs.PlateCarree())
ax4.yaxis.set_major_formatter(cticker.LatitudeFormatter())
# Adjust the fontsize of xticks and yticks
ax4.tick_params(axis='x', labelsize=12)
ax4.tick_params(axis='y', labelsize=12)
# Titles
ax4.set_title("d) STD", loc='left', fontsize=14)
ax4.set_title("SEAS5", loc='right', fontsize=14)
# Adjust the space between subplots
#plt.subplots_adjust(wspace=0.15)

# # Colorbar for SEAS5 STD data
# cbar_ax3 = fig.add_axes([0.12, 0.07, 0.35, 0.02])  # [left, bottom, width, height]
# cbar3 = fig.colorbar(cs3, cax=cbar_ax3, orientation='horizontal')
# cbar3.ax.tick_params(labelsize=14) 
# cbar3.ax.set_xlabel('Temperature (°C)', fontsize=14)

# Colorbar for SEAS5 STD data
#0.2, 0.05, 0.6, 0.01
cbar_ax4 = fig.add_axes([0.2, 0.07, 0.6, 0.04])  # [left, bottom, width, height]
cbar4 = fig.colorbar(cs4, cax=cbar_ax4, orientation='horizontal', shrink=0.8, pad=0.8)
cbar4.ax.tick_params(labelsize=14) 
cbar4.ax.set_xlabel('Temperature (°C)', fontsize=14)
plt.savefig('figure2abcd.png', dpi = 300)
plt.show()
#"""
