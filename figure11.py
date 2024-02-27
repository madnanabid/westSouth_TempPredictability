# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:18:39 2024

@author: Irfan
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
dir1 ="Data path"
dir2 = "Data path"

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

panom=[]
panom=np.sqrt(signal/(signal+noise))
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
meanlan=np.mean(lanina_val)
lan_amp=np.mean(np.sqrt(lanina_val*lanina_val))
meaneln=np.mean(elnino_val)

sst_elnino = panom[ts_nino34>=0.5,:,:] 
sst_lanina = panom[ts_nino34<=0.5*-1.,:,:]

#####Next the SST composite is defined###
elnino_mean= np.mean(sst_elnino,axis=0)
lanina_mean= np.mean(sst_lanina,axis=0)
# ######################Significance Test
df =len(elnino_val)-2
#print(df)
numt=[]
numt=elnino_mean
#print(numt.shape)
denmt=[]
denmt=np.sqrt(np.mean(pow(sst_elnino,2)))/np.sqrt(df)
#print(denmt.shape)
tscore = []
tscore = abs(numt/denmt)
#print(tscore)
#tscore1 = tscore
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025, df)
# print(t90)
# print(t95)

#############El Nino#####################
fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())

ax1.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

data1 = elnino_mean
#print(data1.shape)
data1, lons = add_cyclic_point(data1, coord=f['lon'])

ax1.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax1.coastlines()
ax1.add_feature(cf.BORDERS)

cs1=ax1.contourf(lons, f['lat'], data1, np.arange(0.1,0.9,0.1), transform = ccrs.PlateCarree(),cmap='YlOrRd',extend='both', fontsize=15 )
# Add the rectangular box to the plot
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))



lonEW=63 ; lonWW=69; latSW=32; latNW=35
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(data1[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("Afghanistan=", np.round(val,2))
lonEW=71 ; lonWW=74; latSW=32; latNW=35
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(data1[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("Pakistan=", np.round(val,2))
lonEW=65 ; lonWW=70; latSW=37; latNW=40
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(data1[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("Uzbekistan=", np.round(val,2))
lonEW=77 ; lonWW=80; latSW=32; latNW=35
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(data1[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("NW India=", np.round(val,2))

#----------------------
lonEW=65 ; lonWW=80; latSW=31; latNW=39
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(data1[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
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
plt.title("a) Potential pred.(SAT)", loc='left', fontsize=14 )
plt.title("(El Niño)", loc='right', fontsize=14 )

########### La Nina

df =len(lanina_val)-2
#print(df)
numt=[]
numt=lanina_mean
#print(numt.shape)
denmt=[]
denmt=np.sqrt(np.mean(pow(sst_lanina,2)))/np.sqrt(df)
#print(denmt.shape)
tscore = []
tscore = abs(numt/denmt)
#print(tscore)
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025, df)
# print(t90)
# print(t95)

ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())

ax2.set_extent((55, 91, 20, 41), crs=ccrs.PlateCarree())

data3 = lanina_mean
#print(data1.shape)

data3, lons = add_cyclic_point(data3, coord=f['lon'])
ax2.add_feature(cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='w'))
ax2.coastlines()
ax2.add_feature(cf.BORDERS)

cs2=ax2.contourf(lons, f['lat'], data3, np.arange(0.1,0.9,0.1), transform = ccrs.PlateCarree(),cmap='YlOrRd',extend='both', fontsize=15 )
# Add the rectangular box to the plot
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))

lonEW=63 ; lonWW=69; latSW=32; latNW=35
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(data3[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("Afghanistan=", np.round(val,2))
lonEW=71 ; lonWW=74; latSW=32; latNW=35
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(data3[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("Pakistan=", np.round(val,2))
lonEW=65 ; lonWW=70; latSW=37; latNW=40
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(data3[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("Uzbekistan=", np.round(val,2))
lonEW=77 ; lonWW=80; latSW=32; latNW=35
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(data3[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("NW India=", np.round(val,2))
#----------------------
lonEW=65 ; lonWW=80; latSW=31; latNW=39
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)
print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(data3[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print("WSA=", np.round(val,2))
data4 = tscore
data4, lons = add_cyclic_point(data4, coord=f['lon'])

plt.contourf(lons, f['lat'], data4[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
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
plt.title("b) Potential pred.(SAT)", loc='left', fontsize=14 )
plt.title("(La Niña)", loc='right', fontsize=14 )
cbar_ax2 = fig.add_axes([0.2, 0.47, 0.6, 0.04])  # [left, bottom, width, height]
cbar2 = fig.colorbar(cs2, cax=cbar_ax2, orientation='horizontal', shrink=0.8, pad=0.8)
cbar2.ax.tick_params(labelsize=14) 
fig.tight_layout()
plt.savefig('figure11ab.png', dpi = 300)
plt.show()
