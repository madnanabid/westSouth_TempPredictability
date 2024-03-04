# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 23:18:57 2024

@author: Dell
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import cartopy.feature as cf
path1='Data Path'
fname1='filename'

path2='Data Path'
fname2='filename'

path3='Data Path'
fname3='filename '

path4='Data Path'
fname4='filename'

#-------------Mean-------------
ncep = xr.open_dataset(path1 + fname1, decode_times=False)
merra = xr.open_dataset(path2 + fname2, decode_times=False)
cru = xr.open_dataset(path3 + fname3, decode_times=False)
cpc = xr.open_dataset(path4 + fname4, decode_times=False)

#-------------STD-------------
mean_ncep=ncep.mean(dim='time')

mean_merra=merra.mean(dim='time')
mean_merra = mean_merra -273.15
mean_cru=cru.mean(dim='time')
mean_cpc=cpc.mean(dim='time')


std_merra=merra.std(dim='time')
std_cru=cru.std(dim='time')
std_cpc=cpc.std(dim='time')



fig = plt.figure(figsize=(12,16))
# Set the axes using the specified map projection
ax1 = fig.add_subplot(4, 2, 1, projection=ccrs.PlateCarree())

#ax1=plt.axes(projection=ccrs.PlateCarree())
ax1.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
# Add cyclic point to data
data1=mean_ncep['air']
data1, lons = add_cyclic_point(data1, coord=ncep['lon'])
ax1.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax1.coastlines()
ax1.add_feature(cf.BORDERS)
#ax1.add_feature(cf.OCEAN, facecolor=ocean)
cmap = plt.cm.RdBu_r
cs1=ax1.contourf(lons, ncep['lat'], data1, np.arange(-10,41,5), transform = ccrs.PlateCarree(),norm=colors.CenteredNorm(), cmap=cmap,extend='both' )

# cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar.ax.tick_params(labelsize=20) 
# cbar.ax.set_title('Temperature(°C)',fontsize=20, loc = "right" )

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
plt.title("a) SAT", loc='left', fontsize=14)
plt.title("NCEP", loc='right', fontsize=14)

#---------------------------------
ax2 = fig.add_subplot(4, 2, 2, projection=ccrs.PlateCarree())
ax2.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
mean_ncep = mean_ncep -273.15
std_ncep=ncep.std(dim='time')
# Add cyclic point to data
data2=std_ncep['air']

data2, lons = add_cyclic_point(data2, coord=ncep['lon'])

ax2.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax2.coastlines()
ax2.add_feature(cf.BORDERS)

cs2=ax2.contourf(lons, ncep['lat'], data2, np.arange(0.2,2,0.2), transform = ccrs.PlateCarree(),cmap='YlOrBr',extend='both' )

# cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar.ax.tick_params(labelsize=15) 
# cbar.ax.set_title('Temperature(°C)',fontsize=20, loc = "right" )
# ax1.add_patch( Rectangle((65, 31),
#                         15, 8,
#                         fc ='none', 
#                         ec ='black',
#                         lw = 5) )
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
#plt.title("c) STD (OBS)", loc='left', fontsize=18, fontweight="bold")
plt.title("e) STD", loc='left', fontsize=14)
plt.title("NCEP", loc='right', fontsize=14)

#-----------------------MERRA-------------------------


ax3 = fig.add_subplot(4, 2, 3, projection=ccrs.PlateCarree())
ax3.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
#ax2=plt.axes(projection=ccrs.PlateCarree())
ax3.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
# Add cyclic point to data
data3=mean_merra['t2m']
data3, lons = add_cyclic_point(data3, coord=merra['lon'])
ax3.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax3.coastlines()
ax3.add_feature(cf.BORDERS)
#ax1.add_feature(cf.OCEAN, facecolor=ocean)
cmap = plt.cm.RdBu_r
cs3=ax3.contourf(lons, merra['lat'], data3, np.arange(-10,41,5), transform = ccrs.PlateCarree(),norm=colors.CenteredNorm(), cmap=cmap,extend='both' )

# cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar.ax.tick_params(labelsize=20) 
# cbar.ax.set_title('Temperature(°C)',fontsize=20, loc = "right" )

# Define the xticks for longitude
ax3.set_xticks(np.arange(55,91,5), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax3.xaxis.set_major_formatter(lon_formatter)
plt.xticks(fontsize=12) 

# Define the yticks for latitude
ax3.set_yticks(np.arange(20,41,5), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax3.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=12) 
plt.title("b) SAT", loc='left', fontsize=14)
plt.title("MERRA2", loc='right', fontsize=14)
plt.tight_layout()


#STD
#---------------------------------
ax4 = fig.add_subplot(4, 2, 4, projection=ccrs.PlateCarree())
ax4.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
#mean_nce = mean_ncep -273.15
std_merra=merra.std(dim='time')
# Add cyclic point to data
data4=std_merra['t2m']

data4, lons = add_cyclic_point(data4, coord=merra['lon'])

ax4.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax4.coastlines()
ax4.add_feature(cf.BORDERS)

cs4=ax4.contourf(lons, merra['lat'], data4, np.arange(0.2,2,0.2), transform = ccrs.PlateCarree(),cmap='YlOrBr',extend='both' )

# cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar.ax.tick_params(labelsize=15) 
# cbar.ax.set_title('Temperature(°C)',fontsize=20, loc = "right" )
# ax1.add_patch( Rectangle((65, 31),
#                         15, 8,
#                         fc ='none', 
#                         ec ='black',
#                         lw = 5) )
# Define the xticks for longitude
ax4.set_xticks(np.arange(55,91,5), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax4.xaxis.set_major_formatter(lon_formatter)
plt.xticks(fontsize=12) 
# Define the yticks for latitude
ax4.set_yticks(np.arange(20,41,5), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax4.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=12) 
#plt.title("c) STD (OBS)", loc='left', fontsize=18, fontweight="bold")
plt.title("f) STD", loc='left', fontsize=14)
plt.title("MERRA2", loc='right', fontsize=14)


#-----------------------CRU-------------------------

ax5 = fig.add_subplot(4, 2, 5, projection=ccrs.PlateCarree())
ax5.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
#ax2=plt.axes(projection=ccrs.PlateCarree())
ax5.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
# Add cyclic point to data
data5=mean_cru['tmp']
data5, lons = add_cyclic_point(data5, coord=cru['lon'])
ax5.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax5.coastlines()
ax5.add_feature(cf.BORDERS)
#ax1.add_feature(cf.OCEAN, facecolor=ocean)
cmap = plt.cm.RdBu_r
cs5=ax5.contourf(lons, cru['lat'], data5, np.arange(-10,41,5), transform = ccrs.PlateCarree(),norm=colors.CenteredNorm(), cmap=cmap,extend='both' )

# cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar.ax.tick_params(labelsize=20) 
# cbar.ax.set_title('Temperature(°C)',fontsize=20, loc = "right" )

# Define the xticks for longitude
ax5.set_xticks(np.arange(55,91,5), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax5.xaxis.set_major_formatter(lon_formatter)
plt.xticks(fontsize=12) 

# Define the yticks for latitude
ax5.set_yticks(np.arange(20,41,5), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax5.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=12) 
plt.title("c) SAT", loc='left', fontsize=14)
plt.title("CRU", loc='right', fontsize=14)
#STD
#---------------------------------
ax6 = fig.add_subplot(4, 2, 6, projection=ccrs.PlateCarree())
ax6.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
#mean_nce = mean_ncep -273.15
std_cru=cru.std(dim='time')
# Add cyclic point to data
data6=std_cru['tmp']

data6, lons = add_cyclic_point(data6, coord=cru['lon'])

ax6.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax6.coastlines()
ax6.add_feature(cf.BORDERS)

cs6=ax6.contourf(lons, cru['lat'], data6, np.arange(0.2,2,0.2), transform = ccrs.PlateCarree(),cmap='YlOrBr',extend='both' )

# cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar.ax.tick_params(labelsize=15) 
# cbar.ax.set_title('Temperature(°C)',fontsize=20, loc = "right" )
# ax1.add_patch( Rectangle((65, 31),
#                         15, 8,
#                         fc ='none', 
#                         ec ='black',
#                         lw = 5) )
# Define the xticks for longitude
ax6.set_xticks(np.arange(55,91,5), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax6.xaxis.set_major_formatter(lon_formatter)
plt.xticks(fontsize=12) 
# Define the yticks for latitude
ax6.set_yticks(np.arange(20,41,5), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax6.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=12) 
#plt.title("c) STD (OBS)", loc='left', fontsize=18, fontweight="bold")
plt.title("g) STD", loc='left', fontsize=14)
plt.title("CRU", loc='right', fontsize=14)

#-----------------------CPC-------------------------

ax7 = fig.add_subplot(4, 2, 7, projection=ccrs.PlateCarree())
ax7.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
#ax2=plt.axes(projection=ccrs.PlateCarree())
ax7.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
# Add cyclic point to data
data7=mean_cpc['tmax']
data7, lons = add_cyclic_point(data7, coord=cpc['lon'])
ax7.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax7.coastlines()
ax7.add_feature(cf.BORDERS)
#ax1.add_feature(cf.OCEAN, facecolor=ocean)
cmap = plt.cm.RdBu_r
cs7=ax7.contourf(lons, cpc['lat'], data7, np.arange(-10,41,5), transform = ccrs.PlateCarree(),norm=colors.CenteredNorm(), cmap=cmap,extend='both' )

# cbar = plt.colorbar(cs7, orientation='horizontal', shrink=0.8)
# cbar.ax.tick_params(labelsize=14) 
# cbar.ax.set_title('Temperature(°C)',fontsize=14, loc = "right" )

# Define the xticks for longitude
ax7.set_xticks(np.arange(55,91,5), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax7.xaxis.set_major_formatter(lon_formatter)
plt.xticks(fontsize=12) 

# Define the yticks for latitude
ax7.set_yticks(np.arange(20,41,5), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax7.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=12) 
plt.title("d) SAT", loc='left', fontsize=14)
plt.title("CPC", loc='right', fontsize=14)

# cbar_ax2 = fig.add_axes([0.4, 0.03, 0.6, 0.03])  # [left, bottom, width, height]
# cbar2 = fig.colorbar(cs7, cax=cbar_ax2, orientation='horizontal', shrink=0.8, pad=0.8)
# cbar2.ax.tick_params(labelsize=14) 
#STD
#---------------------------------
ax8 = fig.add_subplot(4, 2, 8, projection=ccrs.PlateCarree())
ax8.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
#mean_nce = mean_ncep -273.15
std_cru=cru.std(dim='time')
# Add cyclic point to data
data8=std_cpc['tmax']

data8, lons = add_cyclic_point(data8, coord=cpc['lon'])

ax8.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax8.coastlines()
ax8.add_feature(cf.BORDERS)

cs8=ax8.contourf(lons, cpc['lat'], data8, np.arange(0.2,2,0.2), transform = ccrs.PlateCarree(),cmap='YlOrBr',extend='both' )

# cbar = plt.colorbar(cs8, orientation='horizontal', shrink=0.8)
# cbar.ax.tick_params(labelsize=14) 
# cbar.ax.set_title('Temperature(°C)',fontsize=14, loc = "right" )
# ax1.add_patch( Rectangle((65, 31),
#                         15, 8,
#                         fc ='none', 
#                         ec ='black',
#                         lw = 5) )
# Define the xticks for longitude
ax8.set_xticks(np.arange(55,91,5), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax8.xaxis.set_major_formatter(lon_formatter)
plt.xticks(fontsize=12) 
# Define the yticks for latitude
ax8.set_yticks(np.arange(20,41,5), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax8.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=12) 
#plt.title("c) STD (OBS)", loc='left', fontsize=18, fontweight="bold")
plt.title("h) STD", loc='left', fontsize=14)
plt.title("CPC", loc='right', fontsize=14)
#plt.tight_layout()

# Adjust layout to make space for colorbars
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.95, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.25)
# Create the first colorbar for SAT
cbar_ax1 = fig.add_axes([0.1, 0.02, 0.4, 0.03])  # [left, bottom, width, height]
cbar1 = fig.colorbar(cs7, cax=cbar_ax1, orientation='horizontal')
cbar1.ax.tick_params(labelsize=14)
cbar1.ax.set_title('Temperature (°C)', fontsize=14)

# Create the second colorbar for STD
cbar_ax2 = fig.add_axes([0.55, 0.02, 0.4, 0.03])  # [left, bottom, width, height]
cbar2 = fig.colorbar(cs8, cax=cbar_ax2, orientation='horizontal')
cbar2.ax.tick_params(labelsize=14)
cbar2.ax.set_title('Temperature (°C)', fontsize=14)


#plt.tight_layout(pad=1.0, hspace=0.2)
plt.savefig('figure_S1.png', dpi = 300)
plt.show()
