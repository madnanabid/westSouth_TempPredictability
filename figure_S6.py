# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 00:33:30 2024

@author: Dell
"""

import numpy as np
from netCDF4 import Dataset, num2date
import sys
from scipy import stats
import random
#import Ngl, Nio
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as m
import mpl_toolkits.basemap as bm

#----------------------CPC
f = Dataset('DataPath +Filename')

g = Dataset('DataPath +Filename')

lonE=65
lonW=80
latS=31
latN=39

timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
tas = f.variables['tmax'][:,:,:]

nt, nlat, nlon = tas.shape
ngrd = nlon*nlat
print(nt)
print(ngrd)

#print(tas)
#this averages over time
msst=np.mean(tas,axis=0)
#print(msst)
#print(msst.shape)
anom=[]
anom=tas-msst
#print(anom.shape)

#here we are defining the index

lon1=np.where(lonvar==lonE)
lon2=np.where(lonvar==lonW)
lat1=np.where(latvar==latS)
lat2=np.where(latvar==latN)
print(lon1,lon2,lat1,lat2)

lonx, latx = np.meshgrid(lonvar, latvar)
weights = np.cos(latx * np.pi / 180.)

tas_avg = np.zeros(nt)

for it in np.arange(nt):
    tas_avg[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
#    tas_avg[it] = np.ma.average(enanom[it, -5:5,190:240], weights=weights[-5:5,190:240])
tas_avg = tas_avg /np.std(tas_avg)

#--------------------------------------------------------
timevar2 = g.variables['time']
lonvar2 = g.variables['lon'][:]
latvar2 = g.variables['lat'][:]
tas2 = g.variables['sst'][:,:,:]
#print(tas)
#this averages over time
msst2=np.mean(tas2,axis=0)
#print(msst)
#print(msst.shape)
#anom=[]
anom2=tas2-msst2

slopes = np.zeros((anom.shape[1],anom.shape[2]))
p_values = np.zeros((anom.shape[1],anom.shape[2]))
corrs = np.zeros((anom.shape[1],anom.shape[2]))



for x in range(0, anom2.shape[1]):
  for y in range(0, anom2.shape[2]):
    punto = anom2[:,x,y]
#    slope, intercept, r_value, p_value, std_err = stats.linregress(ts_nino34,punto)
    slope, intercept, r_value, p_value, std_err = stats.linregress(tas_avg,punto)
    slopes[x,y] = slope
    p_values[x,y] = p_value
    corrs[x,y] = r_value
#print(slope)
#print(p_value)

####  T test for statistical signifcance test ######
df =len(tas_avg)-2
#print(df)
numt=[]
numt=corrs[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(corrs,2))
tscore = []

tscore = (numt/denmt)
#print(tscore.shape)
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)

print(t90)
print(t95)
############# t test end here ####

lat1=np.min(latvar2)
lat2=np.max(latvar2)
lon1=np.min(lonvar2)
lon2=np.max(lonvar2)

[lonall, latall] = np.meshgrid(lonvar2[:], latvar2[:])
#plt.figure(num=None, figsize=(8+4, 8+4), dpi=120, facecolor='w', edgecolor='k')
fig = plt.figure(figsize=(12, 6))
ax4 = fig.add_subplot(2, 2, 1)
#mapproj = bm.Basemap(projection='cyl',llcrnrlat=lat1, llcrnrlon=lon1,urcrnrlat=lat2, urcrnrlon=lon2, lon_0=-80, lat_0=0, resolution='l')

m = bm.Basemap(projection='cyl',llcrnrlat=-40, llcrnrlon=0,urcrnrlat=40, urcrnrlon=361, lon_0=-80, lat_0=0, resolution='l')

#m.drawcoastlines()
m.drawcoastlines(linewidth=0.25)		# Draw coastal boundaries
#m.fillcontinents(color='white',lake_color='white',zorder=2)
m.fillcontinents(color='gray',lake_color='white',zorder=2)

parallels = np.arange(-40,41,20.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-180,180,60.) # make longitude lines every 5 degrees from 95W to 70W
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=12,linewidth=0.0)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12,linewidth=0.0)


x, y = m(lonall, latall)

levels=[-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6 ]
jjj1=m.contourf(x,y,slopes[:,:], levels, extend = 'both', cmap=plt.cm.RdBu_r)
#cs = plt.contourf(x, y, slopes[:,:], levels=levels,  cmap='gray', extend='both', alpha=0)

plt.contourf(x,y,tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['...',None,None, None, '...'],alpha=0)
plt.title("a) Reg.(SATI,SST)", loc='left', fontsize=14)
plt.title("CPC", loc='right', fontsize=14)


#-------------MERRA2

f = Dataset('DataPath +Filename')

g = Dataset('DataPath +Filename')

lonE=65
lonW=80
latS=31
latN=39

timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
tas = f.variables['t2m'][:,:,:]

nt, nlat, nlon = tas.shape
ngrd = nlon*nlat
print(nt)
print(ngrd)

#print(tas)
#this averages over time
msst=np.mean(tas,axis=0)
#print(msst)
#print(msst.shape)
anom=[]
anom=tas-msst
#print(anom.shape)

#here we are defining the index

lon1=np.where(lonvar==lonE)
lon2=np.where(lonvar==lonW)
lat1=np.where(latvar==latS)
lat2=np.where(latvar==latN)
print(lon1,lon2,lat1,lat2)

lonx, latx = np.meshgrid(lonvar, latvar)
weights = np.cos(latx * np.pi / 180.)

tas_avg = np.zeros(nt)

for it in np.arange(nt):
    tas_avg[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
#    tas_avg[it] = np.ma.average(enanom[it, -5:5,190:240], weights=weights[-5:5,190:240])
tas_avg = tas_avg /np.std(tas_avg)

#--------------------------------------------------------
timevar2 = g.variables['time']
lonvar2 = g.variables['lon'][:]
latvar2 = g.variables['lat'][:]
tas2 = g.variables['sst'][:,:,:]
#print(tas)
#this averages over time
msst2=np.mean(tas2,axis=0)
#print(msst)
#print(msst.shape)
#anom=[]
anom2=tas2-msst2

slopes = np.zeros((anom.shape[1],anom.shape[2]))
p_values = np.zeros((anom.shape[1],anom.shape[2]))
corrs = np.zeros((anom.shape[1],anom.shape[2]))

for x in range(0, anom2.shape[1]):
  for y in range(0, anom2.shape[2]):
    punto = anom2[:,x,y]
#    slope, intercept, r_value, p_value, std_err = stats.linregress(ts_nino34,punto)
    slope, intercept, r_value, p_value, std_err = stats.linregress(tas_avg,punto)
    slopes[x,y] = slope
    p_values[x,y] = p_value
    corrs[x,y] = r_value
#print(slope)
#print(p_value)

####  T test for statistical signifcance test ######
df =len(tas_avg)-2
#print(df)
numt=[]
numt=corrs[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(corrs,2))
tscore = []

tscore = (numt/denmt)
#print(tscore.shape)
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)

print(t90)
print(t95)
############# t test end here ####

lat1=np.min(latvar2)
lat2=np.max(latvar2)
lon1=np.min(lonvar2)
lon2=np.max(lonvar2)


#plt.figure(num=None, figsize=(8+4, 8+4), dpi=120, facecolor='w', edgecolor='k')
ax2 = fig.add_subplot(2, 2, 3)
[lonall, latall] = np.meshgrid(lonvar2[:], latvar2[:])
m = bm.Basemap(projection='cyl',llcrnrlat=-40, llcrnrlon=0,urcrnrlat=40, urcrnrlon=361, lon_0=-80, lat_0=0, resolution='l')

#m.drawcoastlines()
m.drawcoastlines(linewidth=0.25)		# Draw coastal boundaries
#m.fillcontinents(color='white',lake_color='white',zorder=2)
m.fillcontinents(color='gray',lake_color='white',zorder=2)

parallels = np.arange(-40,41,20.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-180,180,60.) # make longitude lines every 5 degrees from 95W to 70W
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=12,linewidth=0.0)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12,linewidth=0.0)


x, y = m(lonall, latall)

levels=[-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6 ]
jjj2=m.contourf(x,y,slopes[:,:], levels, extend = 'both', cmap=plt.cm.RdBu_r)
# cs = plt.contourf(x, y, slopes[:,:], levels=levels,  cmap='gray', extend='both', alpha=0)

plt.contourf(x,y,tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['...',None,None, None, '...'],alpha=0)
plt.title("b) Reg.(SATI,SST)", loc='left', fontsize=14)
plt.title("MERRA2", loc='right', fontsize=14)

#-------------CRU

f = Dataset('DataPath +Filename')

g = Dataset('DataPath +Filename')

lonE=65
lonW=80
latS=31
latN=39

timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
tas = f.variables['tmp'][:,:,:]

nt, nlat, nlon = tas.shape
ngrd = nlon*nlat
print(nt)
print(ngrd)

#print(tas)
#this averages over time
msst=np.mean(tas,axis=0)
#print(msst)
#print(msst.shape)
anom=[]
anom=tas-msst
#print(anom.shape)

#here we are defining the index

lon1=np.where(lonvar==lonE)
lon2=np.where(lonvar==lonW)
lat1=np.where(latvar==latS)
lat2=np.where(latvar==latN)
print(lon1,lon2,lat1,lat2)

lonx, latx = np.meshgrid(lonvar, latvar)
weights = np.cos(latx * np.pi / 180.)

tas_avg = np.zeros(nt)

for it in np.arange(nt):
    tas_avg[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
#    tas_avg[it] = np.ma.average(enanom[it, -5:5,190:240], weights=weights[-5:5,190:240])
tas_avg = tas_avg /np.std(tas_avg)

#--------------------------------------------------------
timevar2 = g.variables['time']
lonvar2 = g.variables['lon'][:]
latvar2 = g.variables['lat'][:]
tas2 = g.variables['sst'][0:41,:,:]
#print(tas)
#this averages over time
msst2=np.mean(tas2,axis=0)
#print(msst)
#print(msst.shape)
#anom=[]
anom2=tas2-msst2

slopes = np.zeros((anom.shape[1],anom.shape[2]))
p_values = np.zeros((anom.shape[1],anom.shape[2]))
corrs = np.zeros((anom.shape[1],anom.shape[2]))



for x in range(0, anom2.shape[1]):
  for y in range(0, anom2.shape[2]):
    punto = anom2[:,x,y]
#    slope, intercept, r_value, p_value, std_err = stats.linregress(ts_nino34,punto)
    slope, intercept, r_value, p_value, std_err = stats.linregress(tas_avg,punto)
    slopes[x,y] = slope
    p_values[x,y] = p_value
    corrs[x,y] = r_value
#print(slope)
#print(p_value)

####  T test for statistical signifcance test ######
df =len(tas_avg)-2
#print(df)
numt=[]
numt=corrs[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(corrs,2))
tscore = []

tscore = (numt/denmt)
#print(tscore.shape)
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)

print(t90)
print(t95)
############# t test end here ####

lat1=np.min(latvar2)
lat2=np.max(latvar2)
lon1=np.min(lonvar2)
lon2=np.max(lonvar2)


#plt.figure(num=None, figsize=(8+4, 8+4), dpi=120, facecolor='w', edgecolor='k')
ax3 = fig.add_subplot(2, 2, 2)
[lonall, latall] = np.meshgrid(lonvar2[:], latvar2[:])
#mapproj = bm.Basemap(projection='cyl',llcrnrlat=lat1, llcrnrlon=lon1,urcrnrlat=lat2, urcrnrlon=lon2, lon_0=-80, lat_0=0, resolution='l')

m = bm.Basemap(projection='cyl',llcrnrlat=-40, llcrnrlon=0,urcrnrlat=40, urcrnrlon=361, lon_0=-80, lat_0=0, resolution='l')

#m.drawcoastlines()
m.drawcoastlines(linewidth=0.25)		# Draw coastal boundaries
#m.fillcontinents(color='white',lake_color='white',zorder=2)
m.fillcontinents(color='gray',lake_color='white',zorder=2)

parallels = np.arange(-40,41,20.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-180,180,60.) # make longitude lines every 5 degrees from 95W to 70W
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=12,linewidth=0.0)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12,linewidth=0.0)


x, y = m(lonall, latall)

levels=[-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6 ]
jjj3=m.contourf(x,y,slopes[:,:], levels, extend = 'both', cmap=plt.cm.RdBu_r)

#cs = plt.contourf(x, y, slopes[:,:], levels=levels,  cmap='gray', extend='both', alpha=0)

plt.contourf(x,y,tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['...',None,None, None, '...'],alpha=0)

plt.title("c) Reg.(SATI,SST)", loc='left', fontsize=14)
plt.title("CRU", loc='right', fontsize=14)

#NCEP
f = Dataset('DataPath +Filename')

g = Dataset('DataPath +Filename')

lonE=65
lonW=80
latS=31
latN=39

timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
tas = f.variables['air'][:,:,:]

nt, nlat, nlon = tas.shape
ngrd = nlon*nlat
print(nt)
print(ngrd)

#print(tas)
#this averages over time
msst=np.mean(tas,axis=0)
#print(msst)
#print(msst.shape)
anom=[]
anom=tas-msst
#print(anom.shape)

#here we are defining the index

lon1=np.where(lonvar==lonE)
lon2=np.where(lonvar==lonW)
lat1=np.where(latvar==latS)
lat2=np.where(latvar==latN)
print(lon1,lon2,lat1,lat2)

lonx, latx = np.meshgrid(lonvar, latvar)
weights = np.cos(latx * np.pi / 180.)

tas_avg = np.zeros(nt)

for it in np.arange(nt):
    tas_avg[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
#    tas_avg[it] = np.ma.average(enanom[it, -5:5,190:240], weights=weights[-5:5,190:240])
tas_avg = tas_avg /np.std(tas_avg)

#--------------------------------------------------------
timevar2 = g.variables['time']
lonvar2 = g.variables['lon'][:]
latvar2 = g.variables['lat'][:]
tas2 = g.variables['sst'][:,:,:]
#print(tas)
#this averages over time
msst2=np.mean(tas2,axis=0)
#print(msst)
#print(msst.shape)
#anom=[]
anom2=tas2-msst2

slopes = np.zeros((anom.shape[1],anom.shape[2]))
p_values = np.zeros((anom.shape[1],anom.shape[2]))
corrs = np.zeros((anom.shape[1],anom.shape[2]))

for x in range(0, anom2.shape[1]):
  for y in range(0, anom2.shape[2]):
    punto = anom2[:,x,y]
#    slope, intercept, r_value, p_value, std_err = stats.linregress(ts_nino34,punto)
    slope, intercept, r_value, p_value, std_err = stats.linregress(tas_avg,punto)
    slopes[x,y] = slope
    p_values[x,y] = p_value
    corrs[x,y] = r_value
#print(slope)
#print(p_value)

####  T test for statistical signifcance test ######
df =len(tas_avg)-2
#print(df)
numt=[]
numt=corrs[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(corrs,2))
tscore = []

tscore = (numt/denmt)
#print(tscore.shape)
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)
# t90 = stats.t.ppf(1-0.10, df)
# t95 = stats.t.ppf(1-0.05,df)
print(t90)
print(t95)
############# t test end here ####

lat1=np.min(latvar2)
lat2=np.max(latvar2)
lon1=np.min(lonvar2)
lon2=np.max(lonvar2)


ax4 = fig.add_subplot(2, 2, 4)
[lonall, latall] = np.meshgrid(lonvar2[:], latvar2[:])

m = bm.Basemap(projection='cyl',llcrnrlat=-40, llcrnrlon=0,urcrnrlat=40, urcrnrlon=361, lon_0=-80, lat_0=0, resolution='l')

#m.drawcoastlines()
m.drawcoastlines(linewidth=0.25)		# Draw coastal boundaries
#m.fillcontinents(color='white',lake_color='white',zorder=2)
m.fillcontinents(color='gray',lake_color='white',zorder=2)

parallels = np.arange(-40,41,20.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-180,180,60.) # make longitude lines every 5 degrees from 95W to 70W
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=12,linewidth=0.0)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12,linewidth=0.0)

x, y = m(lonall, latall)

levels=[-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6 ]
jjj4=m.contourf(x,y,slopes[:,:], levels, extend = 'both', cmap=plt.cm.RdBu_r)

#cs = plt.contourf(x, y, slopes[:,:], levels=levels,  cmap='gray', extend='both', alpha=0)

plt.contourf(x,y,tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['...',None,None, None, '...'],alpha=0)

plt.title("d) Reg.(SATI,SST)", loc='left', fontsize=14)
plt.title("NCEP", loc='right', fontsize=14)

fig.tight_layout()

cbar_ax4 = fig.add_axes([0.2, 0.08, 0.6, 0.05])  # [left, bottom, width, height]
cbar2 = fig.colorbar(jjj4, cax=cbar_ax4, orientation='horizontal', shrink=0.8, pad=0.8)
cbar2.ax.tick_params(labelsize=14) 
plt.savefig('figure_S6_abcd.png', dpi = 300)
