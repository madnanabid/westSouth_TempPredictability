# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:57:27 2024

"""

import numpy as np
from netCDF4 import Dataset
from scipy import stats
import matplotlib.pyplot as plt
import mpl_toolkits.basemap as bm
from matplotlib.patches import Rectangle
#------------------SATI vs Z200 ERA5--------------------
f = Dataset('Data Path +file')
g = Dataset('Data Path +file')
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
tas_avg = tas_avg /np.std(tas_avg)

#--------------------------------------------------------
timevar2 = g.variables['time']
levvar2 = g.variables['level'][:]
lonvar2 = g.variables['lon'][:]
latvar2 = g.variables['lat'][:]

tas2 = g.variables['z'][:,4,:,:]
print(levvar2)
#this averages over time
msst2=np.mean(tas2,axis=0)
#print(msst)
#print(msst.shape)
#anom=[]
anom2=(tas2-msst2)/10

slopes = np.zeros((anom.shape[1],anom.shape[2]))
p_values = np.zeros((anom.shape[1],anom.shape[2]))
corrs = np.zeros((anom.shape[1],anom.shape[2]))
for x in range(0, anom2.shape[1]):
  for y in range(0, anom2.shape[2]):
    punto = anom2[:,x,y]
    slope, intercept, r_value, p_value, std_err = stats.linregress(tas_avg,punto)
    slopes[x,y] = slope
    p_values[x,y] = p_value
    corrs[x,y] = r_value
#print(slope)
#print(p_value)

#--------------------------------
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(2, 2, 1)
#domain definition plotting
lat1=10
lat2=61
lon1=10
lon2=121
[lonall, latall] = np.meshgrid(lonvar[:], latvar[:])
mapproj = bm.Basemap(projection='cyl',llcrnrlat=lat1, llcrnrlon=lon1,urcrnrlat=lat2, urcrnrlon=lon2, lon_0=-180, lat_0=0, resolution='l')
mapproj.drawcoastlines(linewidth=0.25)		# Draw coastal boundaries
mapproj.drawcountries(linewidth=0.25)	
x, y = mapproj(lonall, latall)
levels=[-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35]
map2=plt.contourf(x,y,slopes[:,:],np.arange(-35,36,5),cmap=plt.cm.RdBu_r,extend='both')
####  T test for statistical signifcance test ######
df =len(tas_avg)-2
#print(df)
numt=[]
numt=corrs[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(corrs,2))
tscore = []

tscore = (numt/denmt)
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)
print(t90)
print(t95)

plt.contourf(x,y,tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)

parallels = np.arange(10,61,10.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(10,120,20.) # make longitude lines every 5 degrees from 95W to 70W
mapproj.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.01)
mapproj.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.01)

plt.title("a) Reg.(SATI,Z200)", loc='left', fontsize=14)
plt.title("ERA5", loc='right', fontsize=14)
#------------------SATI vs Z200 SEAS5--------------------
dir1= 'Data Path'
dir2= 'Data Path'

bslope = np.zeros((25,181,360))
bcorrs = np.zeros((25,181,360))


count=0
for i in range(25):
#    print(i)
    i+=1
    print(i)
    m2 = Dataset(dir1+'sys5_e'+str(i)+'.nc')
    d1 = Dataset(dir2+'sys5_e'+str(i)+'.nc')
    time = m2.variables['time']
    lon = m2.variables['lon'][:]
    lat = m2.variables['lat'][::-1]  
 
    lev = m2.variables['lev'][:]
#    print(lev)
    may = m2.variables['m1'][:,6,:,:]
    jun = m2.variables['m2'][:,6,:,:]
    mj= (may+jun)/(2*10)


    lonT = d1.variables['lon'][:]
    latT = d1.variables['lat'][:]  
    may1 = d1.variables['m1'][:,:,:]
    jun1 = d1.variables['m2'][:,:,:]
    mj1= (may1+jun1)/2
    nt, nlat, nlon = mj1.shape
    ngrd = nlon*nlat

    lonE=65 ; lonW=80; latS=31; latN=39
    lon1=np.where(lonT==lonE)
    lon2=np.where(lonT==lonW)
    lat1=np.where(latT==latS)
    lat2=np.where(latT==latN)
    msst=np.mean(mj1,axis=0)
    print(lon1,lon2,lat1,lat2)
#    print(msst.shape)

    anom=[]
    anom = mj1-msst
    lonx, latx = np.meshgrid(lonT, latT)
    weights = np.cos(latx * np.pi / 180.)
    tas_avg = np.zeros(nt)

    for it in np.arange(nt):
       tas_avg[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])

    tas_avg = tas_avg /np.std(tas_avg)
  
    panom=[]
    panom =((may+jun)/(2*10))-np.mean((may+jun)/(2*10),axis=0)
    for x in range(0, panom.shape[1]):
        for y in range(0, panom.shape[2]):
            punto = panom[:,x,y]
            slope, intercept, r_value, p_value, std_err = stats.linregress(tas_avg,punto)
            bslope[count,x,y] = slope
            bcorrs[count,x,y] = r_value
    count+=1

#print(bslope.shape)
bsst=np.mean(bslope,axis=0)
acorr1=np.mean(bcorrs,axis=0)
#--------------------------------
ax2 = fig.add_subplot(2, 2, 2)
#domain definition plotting
lat1=10
lat2=61
lon1=10
lon2=121
[lonall, latall] = np.meshgrid(lon[:], lat[:])
mapproj = bm.Basemap(projection='cyl',llcrnrlat=lat1, llcrnrlon=lon1,urcrnrlat=lat2, urcrnrlon=lon2, lon_0=-180, lat_0=0, resolution='l')
mapproj.drawcoastlines(linewidth=0.25)		
mapproj.drawcountries(linewidth=0.25)	
x, y = mapproj(lonall, latall)
levels=[-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35]
map2=plt.contourf(x,y,bsst[:,:],np.arange(-35,36,5),cmap=plt.cm.RdBu_r,extend='both')
####  T test for statistical signifcance test ######
df =23
#print(df)
numt=[]
numt=acorr1[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(acorr1,2))
tscore = []

tscore = (numt/denmt)

t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)
print(t90)
print(t95)

plt.contourf(x,y,tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)

parallels = np.arange(10,61,10.) 
meridians = np.arange(10,120,20.) 
mapproj.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.01)
mapproj.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.01)

plt.title("d) Reg.(SATI,Z200)", loc='left', fontsize=14)
plt.title("SEAS5", loc='right', fontsize=14)

#------------------ NINO34 vs Z200 ERA5----------------------

f = Dataset('Data Path +file')
g = Dataset('Data Path +file')

lonE=190
lonW=240
latS=-5
latN=5

timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
tas = f.variables['sst'][:,:,:]

nt, nlat, nlon = tas.shape
ngrd = nlon*nlat
print(nt)
print(ngrd)

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
tas_avg = tas_avg /np.std(tas_avg)
#--------------------------------------------------------
timevar2 = g.variables['time']
levvar2 = g.variables['level'][:]
lonvar2 = g.variables['lon'][:]
latvar2 = g.variables['lat'][:]

tas2 = g.variables['z'][:,4,:,:]
print(levvar2)
#this averages over time
msst2=np.mean(tas2,axis=0)
#print(msst)
#print(msst.shape)
#anom=[]
anom2=(tas2-msst2)/10

slopes = np.zeros((anom.shape[1],anom.shape[2]))
p_values = np.zeros((anom.shape[1],anom.shape[2]))
corrs = np.zeros((anom.shape[1],anom.shape[2]))

for x in range(0, anom2.shape[1]):
  for y in range(0, anom2.shape[2]):
    punto = anom2[:,x,y]
    slope, intercept, r_value, p_value, std_err = stats.linregress(tas_avg,punto)
    slopes[x,y] = slope
    p_values[x,y] = p_value
    corrs[x,y] = r_value
#print(slope)
#print(p_value)
#--------------------------------
#domain definition plotting
lat1=10
lat2=61
lon1=10
lon2=121
[lonall, latall] = np.meshgrid(lonvar[:], latvar[:])
ax3 = fig.add_subplot(2, 2, 3)
mapproj = bm.Basemap(projection='cyl',llcrnrlat=lat1, llcrnrlon=lon1,urcrnrlat=lat2, urcrnrlon=lon2, lon_0=-180, lat_0=0, resolution='l')
mapproj.drawcoastlines(linewidth=0.25)	
mapproj.drawcountries(linewidth=0.25)	
x, y = mapproj(lonall, latall)
levels=[-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35]
map2=plt.contourf(x,y,slopes[:,:],np.arange(-35,36,5),cmap=plt.cm.RdBu_r,extend='both')
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

plt.contourf(x,y,tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)

parallels = np.arange(10,61,10.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(10,120,20.) # make longitude lines every 5 degrees from 95W to 70W
mapproj.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.01)
mapproj.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.01)

plt.title("b) Reg.(NINO34,Z200)", loc='left', fontsize=14)
plt.title("ERA5", loc='right', fontsize=14)

#------------------ NINO34 vs Z200 SEAS5----------------------
dir1= 'Data Path'
dir2= 'Data Path'

bslope = np.zeros((25,181,360))
bcorrs = np.zeros((25,181,360))


count=0
for i in range(25):
#    print(i)
    i+=1
    print(i)
    m2 = Dataset(dir1+'sys5_e'+str(i)+'.nc')
    d1 = Dataset(dir2+'sys5_e'+str(i)+'.nc')
    time = m2.variables['time']
    lon = m2.variables['lon'][:]
  
    lat = m2.variables['lat'][::-1]  
#    print(lat)
    
    lev = m2.variables['lev'][:]
#    print(lev)
    may = m2.variables['m1'][:,6,::-1,:]
    jun = m2.variables['m2'][:,6,::-1,:]
    mj= (may+jun)/(2*10)

    lonT = d1.variables['lon'][:]
    latT = d1.variables['lat'][:]  
    may1 = d1.variables['m1'][:,:,:]
    jun1 = d1.variables['m2'][:,:,:]
    mj1= (may1+jun1)/2
    nt, nlat, nlon = mj1.shape
    ngrd = nlon*nlat
    lonE=190 ; lonW=240; latS=-5; latN=5
    lon1=np.where(lonT==lonE)
    lon2=np.where(lonT==lonW)
    lat1=np.where(latT==latS)
    lat2=np.where(latT==latN)
    msst=np.mean(mj1,axis=0)
    print(lon1,lon2,lat1,lat2)
#    print(msst.shape)

    anom=[]
    anom = mj1-msst
    lonx, latx = np.meshgrid(lonT, latT)
    weights = np.cos(latx * np.pi / 180.)
    tas_avg = np.zeros(nt)

    for it in np.arange(nt):
       tas_avg[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
   #    tas_avg[it] = np.ma.average(enanom[it, -5:5,190:240], weights=weights[-5:5,190:240])
#       print(tas_avg.shape)
    #tas_avg1 = tas_avg 
    tas_avg = tas_avg /np.std(tas_avg)
    panom=[]
    panom =((may+jun)/(2*10))-np.mean((may+jun)/(2*10),axis=0)
    for x in range(0, panom.shape[1]):
        for y in range(0, panom.shape[2]):
            punto = panom[:,x,y]
            slope, intercept, r_value, p_value, std_err = stats.linregress(tas_avg,punto)
            bslope[count,x,y] = slope
            bcorrs[count,x,y] = r_value
    count+=1

#print(bslope.shape)
bsst=np.mean(bslope,axis=0)
acorr1=np.mean(bcorrs,axis=0)


#--------------------------------
ax4 = fig.add_subplot(2, 2, 4)
#domain definition plotting
lat1=10
lat2=61
lon1=10
lon2=121
[lonall, latall] = np.meshgrid(lonvar[:], latvar[:])
mapproj = bm.Basemap(projection='cyl',llcrnrlat=lat1, llcrnrlon=lon1,urcrnrlat=lat2, urcrnrlon=lon2, lon_0=-180, lat_0=0, resolution='l')
mapproj.drawcoastlines(linewidth=0.25)		# Draw coastal boundaries
mapproj.drawcountries(linewidth=0.25)	

x, y = mapproj(lonall, latall)

levels=[-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35]

map3=plt.contourf(x,y,bsst[:,:],np.arange(-35,36,5),cmap=plt.cm.RdBu_r,extend='both')

####  T test for statistical signifcance test ######
df =1050-2
#print(df)
numt=[]
numt=acorr1[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(acorr1,2))
tscore = []

tscore = (numt/denmt)
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)
print(t90)
print(t95)

plt.contourf(x,y,tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)

parallels = np.arange(10,61,10.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(10,120,20.) # make longitude lines every 5 degrees from 95W to 70W
mapproj.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.01)
mapproj.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.01)

plt.title("e) Reg.(NINO34,Z200)", loc='left', fontsize=14)
plt.title("SEAS5", loc='right', fontsize=14)
plt.tight_layout(pad=2.0,h_pad=0.1,w_pad=2)
# Create a colorbar for the first two subplots
cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.05])  # [left, bottom, width, height]

cbar = plt.colorbar(map3, cax=cbar_ax, orientation='horizontal', shrink=1)
cbar.ax.tick_params(labelsize=14) 
#fig.tight_layout()

plt.savefig('figure5abde.png', dpi = 300)
