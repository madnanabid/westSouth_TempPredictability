# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:24:05 2024
@author: Irfan

"""

import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import cartopy.feature as cf
from scipy import stats
from matplotlib.patches import Rectangle
path1='Data path'
path2='Data Path'
fname1='Data Path'
fname2='Data Path'

f=xr.open_dataset(path1+fname1, decode_times=False)
g=xr.open_dataset(path2+fname2)

timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
tas = f.variables['t2m'][:,:,:]
nt, nlat, nlon = tas.shape
ngrd = nlon*nlat


#print(tas)
#this averages over time
msst=np.mean(tas,axis=0)
#print(msst)
#print(msst.shape)
anom=[]
anom=tas-msst
print(anom.shape)
lonx, latx = np.meshgrid(lonvar, latvar)
weights = np.cos(latx * np.pi / 180.)

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

t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)
print(t90)
print(t95)
fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax1.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
cor, lons = add_cyclic_point(cor, coord=f['lon'])

ax1.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax1.coastlines()
ax1.add_feature(cf.BORDERS)

cs1=ax1.contourf(lons, f['lat'], cor, np.arange(0.1,0.9,0.1), transform = ccrs.PlateCarree(),cmap='YlOrRd',extend='both', fontsize=15 )
ax1.add_patch(Rectangle((63, 32),
                        6, 3,
                        fc ='none', 
                        ec ='black',
                        linestyle='dashed',
                        lw = 2) )
lonEW=63 ; lonWW=69; latSW=32; latNW=35
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)

print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(cor[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("Afghanistan=", np.round(val,2))
ax1.add_patch(Rectangle((71, 32),
                        3, 3,
                        fc ='none', 
                        ec ='black',
                        linestyle='dashed',
                        lw = 2) )
lonEW=71 ; lonWW=74; latSW=32; latNW=35
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(cor[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("Pakistan=", np.round(val,2))

ax1.add_patch(Rectangle((65, 37),
                        5, 3,
                        fc ='none', 
                        ec ='black',
                        linestyle='dashed',
                        lw = 2) )
lonEW=65 ; lonWW=70; latSW=37; latNW=40
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(cor[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("Uzbekistan=", np.round(val,2))
ax1.add_patch(Rectangle((77, 32),
                        3.5, 3.5,
                        fc ='none', 
                        ec ='black',
                        linestyle='dashed',
                        lw = 2) )
lonEW=77 ; lonWW=80; latSW=32; latNW=35
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(cor[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("NW India=", np.round(val,2))

lonEW=65 ; lonWW=80; latSW=31; latNW=39
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(cor[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("WSA=", np.round(val,2))

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

plt.title("a) Actual skill(SAT)", loc='left', fontsize=14 )
#-------------- Potential Predictability-------------------
# Data path
diri = "Data path"

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
rlimit=np.sqrt(signal_variance/(signal_variance + noise_variance))
#print(noise.shape)
#print(signal_variance.shape)

ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
ax2.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())
ax2.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

# Add cyclic point to data
data2 = rlimit
#print(data1.shape)

data2, lons = add_cyclic_point(data2, coord=data.variables['lon'][:])

ax2.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax2.coastlines()
ax2.add_feature(cf.BORDERS)
cs2 = ax2.contourf(lons, data.variables['lat'][:], data2, np.arange(0.1, 0.9, 0.1), transform=ccrs.PlateCarree(), cmap='YlOrRd', extend='both', fontsize=15)
ax2.add_patch(Rectangle((63, 32),
                        6, 3,
                        fc ='none', 
                        ec ='black',
                        linestyle='dashed',
                        lw = 2) )
lonEW=63 ; lonWW=69; latSW=32; latNW=35
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
#msst=np.mean(nd1,axis=0)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(rlimit[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("Afghanistan=", np.round(val,2))
ax2.add_patch(Rectangle((71, 32),
                        3, 3,
                        fc ='none', 
                        ec ='black',
                        linestyle='dashed',
                        lw = 2) )
lonEW=71 ; lonWW=74; latSW=32; latNW=35
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(rlimit[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("Pakistan=", np.round(val,2))

ax2.add_patch(Rectangle((65, 37),
                        5, 3,
                        fc ='none', 
                        ec ='black',
                        linestyle='dashed',
                        lw = 2) )
lonEW=65 ; lonWW=70; latSW=37; latNW=40
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
#msst=np.mean(nd1,axis=0)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(rlimit[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("Uzbekistan=", np.round(val,2))
ax2.add_patch(Rectangle((77, 32),
                        3.5, 3.5,
                        fc ='none', 
                        ec ='black',
                        linestyle='dashed',
                        lw = 2) )
lonEW=77 ; lonWW=80; latSW=32; latNW=35
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
#msst=np.mean(nd1,axis=0)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(rlimit[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("NW India=", np.round(val,2))

#----------------------
lonEW=65 ; lonWW=80; latSW=31; latNW=39
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
#msst=np.mean(nd1,axis=0)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(rlimit[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("WSA=", np.round(val,2))
df =len(ensemble_mean)-2
print(df)
numt=[]
numt=data2*np.sqrt(df)
#print(numt.shape)
denmt=[]
denmt=np.sqrt(1-pow(data2,2))
#print(denmt.shape)
tscore = []
tscore = abs(numt/denmt)
#print(tscore)
#tscore1 = tscore
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025, df)
ax2.set_xticks(np.arange(55,91,5), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)
plt.xticks(fontsize=12) 

# Define the yticks for latitude
ax2.set_yticks(np.arange(20,41,5), crs=ccrs.PlateCarree())

lat_formatter = cticker.LatitudeFormatter()
ax2.yaxis.set_major_formatter(lat_formatter)
plt.yticks(fontsize=12) 
plt.contourf(lons, data.variables['lat'][:], tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)
plt.title("b) Potential pred.(SAT)", loc='left', fontsize=14 )
cbar_ax2 = fig.add_axes([0.2, 0.47, 0.6, 0.04])  # [left, bottom, width, height]
cbar2 = fig.colorbar(cs2, cax=cbar_ax2, orientation='horizontal', shrink=0.8, pad=0.8)
cbar2.ax.tick_params(labelsize=14) 
fig.tight_layout()
plt.savefig('figure9ab.png', dpi = 300)
plt.show()
