# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:24:05 2024

@author: Dell
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import cartopy.feature as cf
from matplotlib.patches import Rectangle

# Data path
diri = "Data Path"

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
    may = data.variables['m1'][:,:,:]
    jun = data.variables['m2'][:,:,:]
    mj= (may+jun)/2
    ensemble_data.append(mj)  

ensemble_data = np.array(ensemble_data)

ensemble_data = np.array(ensemble_data)

#print(ensemble_data.shape)

# Calculate ensemble mean
ensemble_mean = np.mean(ensemble_data, axis=0)
print(ensemble_mean.shape)
nt, nlat, nlon = ensemble_mean.shape
ngrd = nlon*nlat
lonx, latx = np.meshgrid(lonvar, latvar)
weights = np.cos(latx * np.pi / 180.)
time_mean = np.mean(ensemble_mean, axis=0)
print(time_mean.shape)
# Calculate noise variance
diff_squared = np.power((ensemble_mean - time_mean), 2)
#print(diff_squared.shape)
signal_variance = np.mean(diff_squared, axis=0)
#print(noise.shape)
print(signal_variance.shape)
diff_squared = np.power((ensemble_data - ensemble_mean), 2)
print(diff_squared.shape)
noise = np.mean(diff_squared, axis=0)
noise_variance = np.mean(noise, axis=0)
print(noise.shape)
fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax1.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

data2 = signal_variance
#print(data1.shape)

data2, lons = add_cyclic_point(data2, coord=data.variables['lon'][:])

ax1.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax1.coastlines()
ax1.add_feature(cf.BORDERS)
cs1 = ax1.contourf(lons, data.variables['lat'][:], data2, np.arange(0.2,1.7,0.2), transform=ccrs.PlateCarree(), cmap='YlOrRd', extend='both', fontsize=15)
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))

ax1.set_xticks(np.arange(55,91,5), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)

plt.xticks(fontsize=12) 

# Define the yticks for latitude
ax1.set_yticks(np.arange(20,41,5), crs=ccrs.PlateCarree())

lat_formatter = cticker.LatitudeFormatter()
ax1.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=12) 

plt.title("a) Signal variance", loc='left', fontsize=14 )
#Noise Variance

ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
ax2.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
ax2.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

data3 = noise_variance
#print(data1.shape)
data3, lons = add_cyclic_point(data3, coord=data.variables['lon'][:])

ax2.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax2.coastlines()
ax2.add_feature(cf.BORDERS)
cs2 = ax2.contourf(lons, data.variables['lat'][:], data3, np.arange(0.2,1.7,0.2), transform=ccrs.PlateCarree(), cmap='YlOrRd', extend='both', fontsize=15)
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))

ax2.set_xticks(np.arange(55,91,5), crs=ccrs.PlateCarree())

lon_formatter = cticker.LongitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)

plt.xticks(fontsize=12) 

# Define the yticks for latitude
ax2.set_yticks(np.arange(20,41,5), crs=ccrs.PlateCarree())

lat_formatter = cticker.LatitudeFormatter()
ax2.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=12) 

plt.title("b) Noise variance", loc='left', fontsize=14 )

cbar_ax2 = fig.add_axes([0.2, 0.47, 0.6, 0.04])  # [left, bottom, width, height]
cbar2 = fig.colorbar(cs2, cax=cbar_ax2, orientation='horizontal', shrink=0.8, pad=0.8)
cbar2.ax.tick_params(labelsize=14) 
fig.tight_layout()
plt.savefig('figure10ab.png', dpi = 300)
plt.show()
