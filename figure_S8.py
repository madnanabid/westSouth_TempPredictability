# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:18:39 2024

@author:
"""


import netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import cartopy.feature as cf
from scipy import stats
from matplotlib.patches import Rectangle
dir1 ="Data Path"
dir2 = "Data Path"

f = xr.open_dataset(dir1+'filename', decode_times=False)

timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
sst = f.variables['sst'][:,:,:]
#mean of the Sea Surface temperatures (SSTs)
msst=np.mean(sst,axis=0)
anom=[]
anom=sst-msst
#print(sst.shape)
nt, nlat, nlon = sst.shape
ngrd = nlon*nlat

# List of ensemble file names
file_names = ["sys5_e1.nc", "sys5_e2.nc", "sys5_e3.nc", "sys5_e4.nc", "sys5_e5.nc",
              "sys5_e6.nc", "sys5_e7.nc", "sys5_e8.nc", "sys5_e9.nc", "sys5_e10.nc",
              "sys5_e11.nc", "sys5_e12.nc", "sys5_e13.nc", "sys5_e14.nc", "sys5_e15.nc",
              "sys5_e16.nc", "sys5_e17.nc", "sys5_e18.nc", "sys5_e19.nc", "sys5_e20.nc",
              "sys5_e21.nc", "sys5_e22.nc", "sys5_e23.nc", "sys5_e24.nc", "sys5_e25.nc"]

# Load ensemble data into an array
ensemble_data = []
for file_name in file_names:
    file_path = dir2 + file_name
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

# Calculate ensemble mean
ensemble_mean = np.mean(ensemble_data, axis=0)
#print(ensemble_mean.shape)

time_mean = np.mean(ensemble_mean, axis=0)
# print(time_mean.shape)
signal = np.power((ensemble_mean - time_mean), 2)
print(signal.shape)
#print(signal_variance.shape)
diff_squared = np.power((ensemble_data - ensemble_mean), 2)
noise = np.mean(diff_squared, axis=0)
print(noise.shape)
#Defining the index
lonE=190
lonW=240
latS=-5
latN=5

lon1=np.where(lonvar==lonE)
lon2=np.where(lonvar==lonW)
lat1=np.where(latvar==latS)
lat2=np.where(latvar==latN)
print(lon1,lon2,lat1,lat2)

#Signal
panom_signal=[]
panom_signal=signal

#Noise

panom_noise=[]
panom_noise=noise
#print(panom.shape)
#Nino34 Index
lonx, latx = np.meshgrid(lonvar, latvar)
weights = np.cos(latx * np.pi / 180.)
ts_nino34 = np.zeros(nt)
for it in np.arange(nt):
    ts_nino34[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    
print(np.round(ts_nino34,2))
std_nino34=np.round(np.std(ts_nino34),2)
lanina_val= ts_nino34[ts_nino34<=-1*0.5]
elnino_val= ts_nino34[ts_nino34>=0.5]
print(lanina_val.shape,elnino_val.shape)

#Signal
sst_elnino_signal = panom_signal[ts_nino34>=0.5,:,:] 
sst_lanina_signal = panom_signal[ts_nino34<=0.5*-1.,:,:]

#####Next the SST composite is defined###
elnino_mean_signal= np.mean(sst_elnino_signal,axis=0)
lanina_mean_signal= np.mean(sst_lanina_signal,axis=0)

#Noise
sst_elnino_noise = panom_noise[ts_nino34>=0.5,:,:] 
sst_lanina_noise = panom_noise[ts_nino34<=0.5*-1.,:,:]

#####Next the SST composite is defined###
elnino_mean_noise= np.mean(sst_elnino_noise,axis=0)
lanina_mean_noise= np.mean(sst_lanina_noise,axis=0)

#############Signal El Nino#####################
fig = plt.figure(figsize=(12, 10))

ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())

ax1.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

data1 = elnino_mean_signal
#print(data1.shape)
data1, lons = add_cyclic_point(data1, coord=f['lon'])

ax1.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax1.coastlines()
ax1.add_feature(cf.BORDERS)

cs1=ax1.contourf(lons, f['lat'], data1, np.arange(0.2,1.7,0.2), transform = ccrs.PlateCarree(),cmap='YlOrRd',extend='both', fontsize=15 )
# Add the rectangular box to the plot
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))
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
plt.title("a) Signal", loc='left', fontsize=14 )
plt.title("(El Niño)", loc='right', fontsize=14 )

########### Signal La Nina


ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())

ax2.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

data3 = lanina_mean_signal
#print(data1.shape)

data3, lons = add_cyclic_point(data3, coord=f['lon'])
ax2.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax2.coastlines()
ax2.add_feature(cf.BORDERS)

cs2=ax2.contourf(lons, f['lat'], data3, np.arange(0.2,1.7,0.2), transform = ccrs.PlateCarree(),cmap='YlOrRd',extend='both', fontsize=15 )
# Add the rectangular box to the plot
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))
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
plt.title("c) Signal", loc='left', fontsize=14 )
plt.title("(La Niña)", loc='right', fontsize=14 )


#############Noise El Nino#####################
ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())

ax3.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

data3 = elnino_mean_noise
#print(data1.shape)
data3, lons = add_cyclic_point(data3, coord=f['lon'])

ax3.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax3.coastlines()
ax3.add_feature(cf.BORDERS)

cs3=ax3.contourf(lons, f['lat'], data3, np.arange(0.2,1.7,0.2), transform = ccrs.PlateCarree(),cmap='YlOrRd',extend='both', fontsize=15 )
# Add the rectangular box to the plot
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))
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
plt.title("b) Noise", loc='left', fontsize=14 )
plt.title("(El Niño)", loc='right', fontsize=14 )

########### Noise La Nina


ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())

ax4.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

data4 = lanina_mean_noise
#print(data1.shape)

data4, lons = add_cyclic_point(data4, coord=f['lon'])
ax4.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax4.coastlines()
ax4.add_feature(cf.BORDERS)

cs4=ax4.contourf(lons, f['lat'], data4, np.arange(0.2,1.7,0.2), transform = ccrs.PlateCarree(),cmap='YlOrRd',extend='both', fontsize=15 )
# Add the rectangular box to the plot
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))
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
plt.title("d) Noise", loc='left', fontsize=14 )
plt.title("(La Niña)", loc='right', fontsize=14 )
cbar_ax4 = fig.add_axes([0.2, 0.04, 0.6, 0.04])  # [left, bottom, width, height]
cbar2 = fig.colorbar(cs4, cax=cbar_ax4, orientation='horizontal', shrink=0.8, pad=0.8)
cbar2.ax.tick_params(labelsize=14) 
fig.tight_layout()
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.95, 
                    top=0.9, 
                    wspace=0.15, 
                    hspace=0.1)
plt.savefig('figure_S8_abcd.png', dpi = 300)
plt.show()
