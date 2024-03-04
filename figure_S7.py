# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:50:24 2024

@author:
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import cartopy.feature as cf
from scipy import stats
from matplotlib.patches import Rectangle
path1='Data Path'

path2='Data Path'
fname1='filename'
fname2='filename'

f=xr.open_dataset(path1+fname1)
g=xr.open_dataset(path2+fname2)


timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
tas = f.variables['air'][:,:,:]
#print(tas)
#this averages over time
msst=np.mean(tas,axis=0)
#print(msst)
#print(msst.shape)
anom=[]
anom=tas-msst
print(anom.shape)

timevar2 = g.variables['time']
lonvar2 = g.variables['lon'][:]
latvar2 = g.variables['lat'][:]
tas2 = g.variables['t2m'][:,:,:]
#print(tas)
#this averages over time
msst2=np.mean(tas2,axis=0)
#print(msst)
#print(msst.shape)
anom2=[]
anom2=tas2-msst2
print(anom2.shape)

xy1 = np.mean(anom*anom2, axis=0)
xx1 = np.mean(anom2*anom2, axis=0)
yy1 = np.mean(anom*anom, axis=0)
den = np.sqrt(xx1*yy1)
cor = xy1/den
print(cor.shape)


df =len(anom2)-2
#print(df)
numt=[]
numt=cor[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(cor,2))
tscore = []
tscore = (numt/denmt)
#print(tscore.shape)
#t90 = stats.t.ppf(1-0.05, df-2)
#t95 = stats.t.ppf(1-0.025,df-2)
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)
print(t90)
print(t95)



fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
#ax1.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

ax1.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

# Add cyclic point to data
# data1 = ds1_mean['tmp'][0,:,:]

cor, lons = add_cyclic_point(cor, coord=f['lon'])

ax1.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax1.coastlines()
ax1.add_feature(cf.BORDERS)

cs1=ax1.contourf(lons, f['lat'], cor, np.arange(0.1,0.9,0.1), transform = ccrs.PlateCarree(),cmap='YlOrRd',extend='both', fontsize=15 )
#cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar.ax.tick_params(labelsize=20)
# ax1.add_patch(Rectangle((63, 32),
#                         6, 3,
#                         fc ='none', 
#                         ec ='black',
#                         linestyle='dashed',
#                         lw = 2) )

# ax1.add_patch(Rectangle((71, 32),
#                         3, 3,
#                         fc ='none', 
#                         ec ='black',
#                         linestyle='dashed',
#                         lw = 2) )
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))
data2 = tscore
data2, lons = add_cyclic_point(data2, coord=f['lon'])

plt.contourf(lons, f['lat'], data2[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)

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

# ticklabs = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(ticklabs, fontsize=20)
#ax1.set_yticklabels(ax1_y, fontsize=15)
plt.title("a) Actual skill(SAT)", loc='left', fontsize=14 )
plt.title("NCEP", loc='right', fontsize=14 )

#--------------------MERRA2--------------------------------------
path1='Data Path'

path2='Data Path'
fname1='filename'
fname2='filename'

f=xr.open_dataset(path1+fname1)
g=xr.open_dataset(path2+fname2)



timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
tas = f.variables['t2m'][:,:,:]
tas = tas-273.15
#print(tas)
#this averages over time
msst=np.mean(tas,axis=0)
#print(msst)
#print(msst.shape)
anom=[]
anom=tas-msst
print(anom.shape)

timevar2 = g.variables['time']
lonvar2 = g.variables['lon'][:]
latvar2 = g.variables['lat'][:]
tas2 = g.variables['t2m'][:,:,:]
#print(tas)
#this averages over time
msst2=np.mean(tas2,axis=0)
#print(msst)
#print(msst.shape)
anom2=[]
anom2=tas2-msst2
print(anom2.shape)

xy1 = np.mean(anom*anom2, axis=0)
xx1 = np.mean(anom2*anom2, axis=0)
yy1 = np.mean(anom*anom, axis=0)
den = np.sqrt(xx1*yy1)
cor = xy1/den
print(cor.shape)


df =len(anom2)-2
#print(df)
numt=[]
numt=cor[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(cor,2))
tscore = []
tscore = (numt/denmt)
#print(tscore.shape)
#t90 = stats.t.ppf(1-0.05, df-2)
#t95 = stats.t.ppf(1-0.025,df-2)
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)
print(t90)
print(t95)



ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())

ax2.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

# Add cyclic point to data
# data1 = ds1_mean['tmp'][0,:,:]

cor, lons = add_cyclic_point(cor, coord=f['lon'])

ax2.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax2.coastlines()
ax2.add_feature(cf.BORDERS)

cs2=ax2.contourf(lons, f['lat'], cor, np.arange(0.1,0.9,0.1), transform = ccrs.PlateCarree(),cmap='YlOrRd',extend='both', fontsize=15 )
#cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar.ax.tick_params(labelsize=20)
# ax1.add_patch(Rectangle((63, 32),
#                         6, 3,
#                         fc ='none', 
#                         ec ='black',
#                         linestyle='dashed',
#                         lw = 2) )

# ax1.add_patch(Rectangle((71, 32),
#                         3, 3,
#                         fc ='none', 
#                         ec ='black',
#                         linestyle='dashed',
#                         lw = 2) )
# ax1.add_patch(Rectangle((65.2, 37.5),
#                         4.2, 2.8,
#                         fc ='none', 
#                         ec ='black',
#                         linestyle='dashed',
#                         lw = 2) )
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))
data2 = tscore
data2, lons = add_cyclic_point(data2, coord=f['lon'])

plt.contourf(lons, f['lat'], data2[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)

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

# ticklabs = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(ticklabs, fontsize=20)
#ax1.set_yticklabels(ax1_y, fontsize=15)
plt.title("b) Actual skill(SAT)", loc='left', fontsize=14 )
plt.title("MERRA2", loc='right', fontsize=14 )

#-----------CRU-----------------------
path1='Data Path'

path2='Data Path'
fname1='filename'
fname2='filename'

f=xr.open_dataset(path1+fname1)
g=xr.open_dataset(path2+fname2)



timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
tas = f.variables['tmp'][:,:,:]
print(tas.shape)
#this averages over time
msst=np.mean(tas,axis=0)
#print(msst)
#print(msst.shape)
anom=[]
anom=tas-msst
#print(anom.shape)

timevar2 = g.variables['time']
lonvar2 = g.variables['lon'][:]
latvar2 = g.variables['lat'][:]
#tas2 = g.variables['t2m'][:,:,:]
tas2 = g.variables['t2m'][0:41,:,:]
print(tas2.shape)
#this averages over time
msst2=np.mean(tas2,axis=0)
#print(msst)
#print(msst.shape)
anom2=[]
anom2=tas2-msst2
print(anom2.shape)

xy1 = np.mean(anom*anom2, axis=0)
xx1 = np.mean(anom2*anom2, axis=0)
yy1 = np.mean(anom*anom, axis=0)
den = np.sqrt(xx1*yy1)
cor = xy1/den
print(cor.shape)


df =len(anom2)-2
#print(df)
numt=[]
numt=cor[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(cor,2))
tscore = []
tscore = (numt/denmt)
#print(tscore.shape)
#t90 = stats.t.ppf(1-0.05, df-2)
#t95 = stats.t.ppf(1-0.025,df-2)
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)
print(t90)
print(t95)



ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())

ax3.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

# Add cyclic point to data
# data1 = ds1_mean['tmp'][0,:,:]

cor, lons = add_cyclic_point(cor, coord=f['lon'])

ax3.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax3.coastlines()
ax3.add_feature(cf.BORDERS)

cs3=ax3.contourf(lons, f['lat'], cor, np.arange(0.1,0.9,0.1), transform = ccrs.PlateCarree(),cmap='YlOrRd',extend='both', fontsize=15 )
#cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar.ax.tick_params(labelsize=20)
# ax1.add_patch(Rectangle((63, 32),
#                         6, 3,
#                         fc ='none', 
#                         ec ='black',
#                         linestyle='dashed',
#                         lw = 2) )

# ax1.add_patch(Rectangle((71, 32),
#                         3, 3,
#                         fc ='none', 
#                         ec ='black',
#                         linestyle='dashed',
#                         lw = 2) )
# ax1.add_patch(Rectangle((65.2, 37.5),
#                         4.2, 2.8,
#                         fc ='none', 
#                         ec ='black',
#                         linestyle='dashed',
#                         lw = 2) )
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))
data3 = tscore
data3, lons = add_cyclic_point(data3, coord=f['lon'])

plt.contourf(lons, f['lat'], data3[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)

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

# ticklabs = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(ticklabs, fontsize=20)
#ax1.set_yticklabels(ax1_y, fontsize=15)
plt.title("c) Actual skill(CRU)", loc='left', fontsize=14 )
plt.title("CRU", loc='right', fontsize=14 )

#----------------------CPC------------------------
path1='Data Path'

path2='Data Path'
fname1='filename'
fname2='filename'

f=xr.open_dataset(path1+fname1, decode_times=False)
g=xr.open_dataset(path2+fname2)

# f = Dataset('E:/DATA/ERA-DATA/ts/MJ/dt.era5_05_tas_8116.MJ.nc')
# g = Dataset('E:/DATA/ECMWF-SYS5/t2m/netcdf/apr_int/test2/dt/ensmean.sys5.t2m.MJ.1981-2016.nc')

timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
tas = f.variables['tmax'][:,:,:]
#tas = tas-273.15
#print(tas)
#this averages over time
msst=np.mean(tas,axis=0)
#print(msst)
#print(msst.shape)
anom=[]
anom=tas-msst
print(anom.shape)

timevar2 = g.variables['time']
lonvar2 = g.variables['lon'][:]
latvar2 = g.variables['lat'][:]
tas2 = g.variables['t2m'][:,:,:]
#print(tas)
#this averages over time
msst2=np.mean(tas2,axis=0)
#print(msst)
#print(msst.shape)
anom2=[]
anom2=tas2-msst2
print(anom2.shape)

xy1 = np.mean(anom*anom2, axis=0)
xx1 = np.mean(anom2*anom2, axis=0)
yy1 = np.mean(anom*anom, axis=0)
den = np.sqrt(xx1*yy1)
cor = xy1/den
print(cor.shape)


df =len(anom2)-2
#print(df)
numt=[]
numt=cor[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(cor,2))
tscore = []
tscore = (numt/denmt)
#print(tscore.shape)
#t90 = stats.t.ppf(1-0.05, df-2)
#t95 = stats.t.ppf(1-0.025,df-2)
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)
print(t90)
print(t95)



ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())

ax4.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

# Add cyclic point to data
# data1 = ds1_mean['tmp'][0,:,:]

cor, lons = add_cyclic_point(cor, coord=f['lon'])

ax4.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax4.coastlines()
ax4.add_feature(cf.BORDERS)

cs4=ax4.contourf(lons, f['lat'], cor, np.arange(0.1,0.9,0.1), transform = ccrs.PlateCarree(),cmap='YlOrRd',extend='both', fontsize=15 )
#cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar = plt.colorbar(cs1, orientation='horizontal', shrink=0.8)
# cbar.ax.tick_params(labelsize=20)
# ax1.add_patch(Rectangle((63, 32),
#                         6, 3,
#                         fc ='none', 
#                         ec ='black',
#                         linestyle='dashed',
#                         lw = 2) )

# ax1.add_patch(Rectangle((71, 32),
#                         3, 3,
#                         fc ='none', 
#                         ec ='black',
#                         linestyle='dashed',
#                         lw = 2) )
# ax1.add_patch(Rectangle((65.2, 37.5),
#                         4.2, 2.8,
#                         fc ='none', 
#                         ec ='black',
#                         linestyle='dashed',
#                         lw = 2) )
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))
data4 = tscore
data4, lons = add_cyclic_point(data4, coord=f['lon'])

plt.contourf(lons, f['lat'], data4[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)

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

# ticklabs = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(ticklabs, fontsize=20)
#ax1.set_yticklabels(ax1_y, fontsize=15)
plt.title("d) Actual skill(SAT)", loc='left', fontsize=14 )
plt.title("CPC", loc='right', fontsize=14 )
fig.tight_layout()
# Adjust layout to make space for colorbars
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.95, 
                    top=0.9, 
                    wspace=0.15, 
                    hspace=0.1)
cbar_ax = fig.add_axes([0.2, 0.035, 0.6, 0.05])  # [left, bottom, width, height]
cbar = plt.colorbar(cs4, cax=cbar_ax, orientation='horizontal', shrink=1)
cbar.ax.tick_params(labelsize=14) 
plt.savefig('figure_S7_abde.png', dpi = 300)
