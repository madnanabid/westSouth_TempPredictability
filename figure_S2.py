# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:50:27 2024

@author:
"""
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import cartopy.feature as cf
from matplotlib.patches import Rectangle
#path1= 'E:/DATA/ECMWF-SYS5/t2m/netcdf/apr_int/test2/'
#fname1 = 'ensmean.sys5.t2m.MJ.1981-2016.nc'
path1= 'Data Path'
fname1 = 'filename'

path2='Data Path'
fname2='filename'

#MOD
ds1=xr.open_dataset(path1+fname1)
ds1 = ds1-273.15
#print(ds)
ds1_mean=ds1.mean(dim='time')
#print(ds1_mean)
# Make the figure larger


#OBS

ds2=xr.open_dataset(path2+fname2, decode_times=False)
ds2 = ds2-273.15
#print(ds1)
ds2_mean=ds2.mean(dim='time')


fig = plt.figure(figsize=(10,8))

# Set the axes using the specified map projection
ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax1.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

# Add cyclic point to data
data_mod = ds1_mean['t2m']
data_obs = ds2_mean['t2m']

data1 = data_mod-data_obs
#print(ds1_mean)
#quit()
data1, lons = add_cyclic_point(data1, coord=ds1['lon'])

ax1.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax1.coastlines()
ax1.add_feature(cf.BORDERS)

cmap = plt.cm.RdBu_r
cs1=ax1.contourf(lons, ds1['lat'], data1, np.arange(-5,5.1,0.5), transform = ccrs.PlateCarree(),norm=colors.CenteredNorm(), cmap=cmap,extend='both' )
#cs1=ax1.contourf(lons, ds1['lat'], data1, np.arange(-40,41,5), transform = ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend='both' )
cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.9, pad=0.2,aspect=13)
cbar.ax.tick_params(labelsize=12) 
cbar.ax.set_title('Temperature(°C)',fontsize=12, loc = "right" )

#ax1.set_xlabel("Temperature(°C)",
#              fontweight ='bold',
#              fontsize=16,
#              loc="right")
# Define the xticks for longitude
ax1.set_xticks(np.arange(55,91,5), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
plt.xticks(fontsize=12) 

# Define the yticks for latitude
ax1.set_yticks(np.arange(20,41,5), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax1.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=12) 
plt.gca().add_patch(Rectangle((65, 31),
                          15, 8,
                          fc ='none', 
                          ec ='black',
                          lw = 2))
plt.title("a) Bias(MOD-OBS)", loc='left', fontsize=14)




#-------------RMSE--------------------
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import cartopy.feature as cf
from matplotlib.patches import Rectangle
path1 = 'Data Path'
fname1 = 'filename'

path2 = 'Data Path'
fname2 = 'filename'

# MOD
ds1 = nc.Dataset(path1 + fname1, "r")
lon = ds1.variables['lon'][:]
lat = ds1.variables['lat'][:]
time = ds1.variables['time']
#tas = f.variables['t2m'][:, :, :]
ds1_t2m = ds1.variables['t2m'][:]
ds1 = ds1_t2m 

# OBS
ds2 = nc.Dataset(path2 + fname2, "r")
ds2_t2m = ds2.variables['t2m'][:]
ds2 = ds2_t2m

print(ds2.shape)

dif_power = np.power((ds1 - ds2), 2)

rmse = np.sqrt(np.mean(dif_power, axis=0))
print(rmse.shape)

#---------------------
ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
plt.subplots_adjust(wspace=0.1)
ax2.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
# ax2 = plt.axes(projection=ccrs.PlateCarree())
# ax2.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

ax2.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax2.coastlines()
ax2.add_feature(cf.BORDERS)
# Add coastlines
ax2.coastlines()

# Plot RMSE
cmap = plt.cm.YlOrRd
cs2 = ax2.contourf(lon, lat, rmse, np.arange(1, 5, 0.5), transform=ccrs.PlateCarree(), cmap=cmap, extend='both')

#cbar_ax2 = fig.add_axes([0.2, 0.47, 0.6, 0.04])  # [left, bottom, width, height]
cbar = plt.colorbar(cs2, orientation='horizontal', shrink=0.9, pad=0.2, aspect=13)
cbar.ax.tick_params(labelsize=12) 
cbar.ax.set_title('Temperature(°C)',fontsize=12, loc = "right" )
plt.gca().add_patch(Rectangle((65, 31),
                          15, 8,
                          fc ='none', 
                          ec ='black',
                          lw = 2))
#ax1.set_xlabel("Temperature(°C)",
#              fontweight ='bold',
#              fontsize=16,
#              loc="right")
# Define the xticks for longitude
ax2.set_xticks(np.arange(55,91,5), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)
plt.xticks(fontsize=12) 

# Define the yticks for latitude
ax2.set_yticks(np.arange(20,41,5), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax2.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=12) 

plt.title("b) RMSE", loc='left', fontsize=14)
plt.title("SEAS5", loc='right', fontsize=14)

fig.tight_layout()
# fig = plt.figure(figsize=(13,8))
# ax = plt.axes(projection=ccrs.PlateCarree())

# Adjust layout to make space for colorbars
plt.savefig('figure_S2.png', dpi = 300)


