# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 00:27:21 2024

"""
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy.stats import linregress

plt.close('all') # this will close any previously opemj matplotlib plot
#fig = plt.figure()
  

dir1 = "Data Path"
dir2 = "Data Path"

tmp = nc.Dataset(dir1+"filename","r")
sst = nc.Dataset(dir2+"filename","r")

############SATI
timevar = tmp.variables['time']
lonvar = tmp.variables['lon'][:]
latvar = tmp.variables['lat'][:]
tas = tmp.variables['t2m'][:,:,:]

nt, nlat, nlon = tas.shape
ngrd = nlon*nlat
print(nt)
print(ngrd)

mtmp=np.mean(tas,axis=0)

anom=[]
anom=tas-mtmp

lonE=65
lonW=80
latS=31
latN=39

lon3=np.where(lonvar==lonE)
lon4=np.where(lonvar==lonW)
lat3=np.where(latvar==latS)
lat4=np.where(latvar==latN)
print(lon3,lon4,lat3,lat4)

lonx, latx = np.meshgrid(lonvar, latvar)
weights = np.cos(latx * np.pi / 180.)

tas_sati = np.zeros(nt)

for it in np.arange(nt):
    tas_sati[it] = np.ma.average(anom[it, int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])

############NINO34
timevar1 = sst.variables['time']
lonvar1 = sst.variables['lon'][:]
latvar1 = sst.variables['lat'][:]
tsst = sst.variables['sst'][:,:,:]
#print(tas)
#this averages over time
msst=np.mean(tsst,axis=0)
#print(msst)
#print(msst.shape)
anomsst=[]
anomsst=tsst-msst

lonES=190
lonWS=240
latSS=-5
latNS=5

lon5=np.where(lonvar1==lonES)
lon6=np.where(lonvar1==lonWS)
lat5=np.where(latvar1==latSS)
lat6=np.where(latvar1==latNS)
print(lon5,lon6,lat5,lat6)

lonx, latx = np.meshgrid(lonvar1, latvar1)
weights = np.cos(latx * np.pi / 180.)

tas_nino34_obs = np.zeros(nt)

for it in np.arange(nt):
    tas_nino34_obs[it] = np.ma.average(anomsst[it, int(lat5[0]):int(lat6[0]),int(lon5[0]):int(lon6[0])], weights=weights[int(lat5[0]):int(lat6[0]),int(lon5[0]):int(lon6[0])])
    
#tas_nino34_obs = tas_nino34_obs /np.std(tas_nino34_obs)
print(tas_nino34_obs)   

fig = plt.figure(figsize=(12, 8))

# Plot CLM ERA data
ax1 = fig.add_subplot(2, 2, 1)
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(tas_nino34_obs,tas_sati)
#print('OBS')
print('Slope1: %.2f' % slope1, 'Intercept1: %.2f' %intercept1,'r value1: %.2f' %r_value1)
print(intercept1)
#cor = data.NINO34obs.corr(data.SATI)
label = "{:.2f}".format(r_value1)
label1 = "{:.2f}".format(slope1)
#print(label1)
plt.scatter(tas_nino34_obs, tas_sati, c ="black")
#m, b = np. polyfit(data.NINO34obs, data.SATI, 1) #m = slope, b=intercept.
#plt.plot(data.NINO34obs, m*data.NINO34obs + b,color="b", lw=0.5) #add line of best fit.
plt.xlim(-2.5,2.5,0.5)
plt.ylim(-4.5,4.5,0.5)
plt.vlines(0, -4.5, 4.5, lw=1, color='g')
plt.hlines(0, -4.5, 4.5, lw=1, color='g')
xmin=[-0.5,0.5]
ymin=[-4.5]
ymax=[4.5]

plt.fill_between(xmin,ymin,ymax,alpha=0.3,color='tab:orange')

plt.title('a) SATI vs ENSO ',loc = 'left', fontsize=14)
plt.title('ERA5',loc = 'right')
plt.xlabel('Niño3.4(°C)')
plt.ylabel('Temperature(°C)')
plt.text(1.1, 4, 'CC= %s'%label, style ='italic',
         fontsize = 9.5, color ="black")
plt.text(1.1, 3.6, 'Slope= %s'%label1, style ='italic',
         fontsize = 9.5, color ="black")
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

###############SEAS5

dir1="Data Path"
dir2="Data Path"

dir3="Data Path"
dir4="Data Path"

tmp_mod = nc.Dataset(dir1+"filename","r")
sst_mod = nc.Dataset(dir2+"filename","r")

tmp_mod_ensmn = nc.Dataset(dir3+"filename","r")
sst_mod_ensmn = nc.Dataset(dir4+"filename","r")

lat     = tmp_mod.variables['lat'][:]
lon     = tmp_mod.variables['lon'][:]
time    = tmp_mod.variables['time'][:]

maye1 = tmp_mod.variables['m1'][:,:,:]
june1 = tmp_mod.variables['m2'][:,:,:]

msste1=(maye1+june1)/2

nt, nlat, nlon = maye1.shape
ngrd = nlon*nlat
anom1=[]
anom1=msste1
#print(anom1.shape)
lonE=65
lonW=80
latS=31
latN=39
lon1=np.where(lon==lonE)
lon2=np.where(lon==lonW)
lat1=np.where(lat==latS)
lat2=np.where(lat==latN)
#print(lon1,lon2,lat1,lat2)

#ts_sati=[]
lonx, latx = np.meshgrid(lon, lat)
weights = np.cos(latx * np.pi / 180.)
ts_sati1 = np.zeros(nt)

#print(ts_sati1.shape)
for it in np.arange(nt):
    ts_sati1[it] = np.ma.average(anom1[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    
  
ts_meanen=ts_sati1

#-----------------------SEAS5 NINO34
lat     = sst_mod.variables['lat'][:]
lon     = sst_mod.variables['lon'][:]
time    = sst_mod.variables['time'][:]

#print(lat.shape,lon.shape,time.shape)
#quit()

maye1 = sst_mod.variables['m1'][:,:,:]
june1 = sst_mod.variables['m2'][:,:,:]

msste1=(maye1+june1)/2
#print(msst.shape)
anomsst1=[]
anomsst1=msste1

lonEs=190
lonWs=240
latSs=-5
latNs=5
lon7=np.where(lon==lonEs)
lon8=np.where(lon==lonWs)
lat7=np.where(lat==latSs)
lat8=np.where(lat==latNs)
print(lon7,lon8,lat7,lat8)

#ts_nino34_e=[]
lonx, latx = np.meshgrid(lon, lat)
weights = np.cos(latx * np.pi / 180.)
ts_nino34_mod = np.zeros(nt)

for it in np.arange(nt):
    ts_nino34_mod[it] = np.ma.average(anomsst1[it, int(lat7[0]):int(lat8[0]),int(lon7[0]):int(lon8[0])], weights=weights[int(lat7[0]):int(lat8[0]),int(lon7[0]):int(lon8[0])])

#ts_meansst=ts_nino34_e1+ts_nino34_e2+ts_nino34_e3+ts_nino34_e4+ts_nino34_e5+ts_nino34_e6+ts_nino34_e7+ts_nino34_e8+ts_nino34_e9+ts_nino34_e10+ts_nino34_e11+ts_nino34_e12+ts_nino34_e13+ts_nino34_e14+ts_nino34_e15+ts_nino34_e16+ts_nino34_e17+ts_nino34_e18+ts_nino34_e19+ts_nino34_e20+ts_nino34_e21+ts_nino34_e22+ts_nino34_e23+ts_nino34_e24+ts_nino34_e25
ts_meanensst=ts_nino34_mod
#print(ts_meanen)
################ ensmean############
tas = tmp_mod_ensmn.variables['t2m'][:,:,:]
nt, nlat, nlon = tas.shape
ngrd = nlon*nlat
print(tas.shape)

mtmp=np.mean(tas,axis=0)
#print(msst)
#print(msst.shape)
anom=[]
anom=tas-mtmp

lonE=65
lonW=80
latS=31
latN=39

lon3=np.where(lon==lonE)
lon4=np.where(lon==lonW)
lat3=np.where(lat==latS)
lat4=np.where(lat==latN)
print(lon3,lon4,lat3,lat4)

lonx, latx = np.meshgrid(lon, lat)
weights = np.cos(latx * np.pi / 180.)

tas_sati = np.zeros(nt)

for it in np.arange(nt):
    tas_sati[it] = np.ma.average(anom[it, int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])


sst = sst_mod_ensmn.variables['sst'][:,:,:]

print(sst.shape)
msst=np.mean(sst,axis=0)
#print(msst)
#print(msst.shape)
anomsst=[]
anomsst=sst-msst

lonEs=190
lonWs=240
latSs=-5
latNs=5

lon5=np.where(lon==lonEs)
lon6=np.where(lon==lonWs)
lat5=np.where(lat==latSs)
lat6=np.where(lat==latNs)
print(lon5,lon6,lat5,lat6)

lonx, latx = np.meshgrid(lon, lat)
weights = np.cos(latx * np.pi / 180.)

tas_nino34 = np.zeros(nt)

for it in np.arange(nt):
    tas_nino34[it] = np.ma.average(anomsst[it, int(lat5[0]):int(lat6[0]),int(lon5[0]):int(lon6[0])], weights=weights[int(lat5[0]):int(lat6[0]),int(lon5[0]):int(lon6[0])])
    
xmin=[-0.5,0.5]
ymin=[-4.5]
ymax=[4.5]
ax2 = fig.add_subplot(2, 2, 2)
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(ts_meanensst,ts_meanen)
# print('MOD')
print('Slope2: %.2f' % slope2, 'Intercept2: %.2f' %intercept2,'r value2: %.2f' %r_value2)
print(intercept2)
label2 = "{:.2f}".format(r_value2)
label3 = "{:.2f}".format(slope2)
#axs[1].scatter(ts_meanensst,ts_meanen, c ="black")

plt.scatter(ts_meanensst,ts_meanen, c ="grey", s=10)

slope3, intercept3, r_value3, p_value3, std_err3 = linregress(tas_nino34,tas_sati)
print('Slope3: %.2f' % slope3, 'Intercept3: %.2f' %intercept3,'r value3: %.2f' %r_value3)
print(intercept3)
label4 = "{:.2f}".format(r_value3)
label5 = "{:.2f}".format(slope3)

plt.scatter(tas_nino34,tas_sati, c ="black")
# m, b = np. polyfit(data.NINO34mod, data.SATImod, 1) #m = slope, b=intercept.
# axs[1].plot(data.NINO34mod, m*data.NINO34mod + b,color="b", lw=0.5) #add line of best fit.
plt.xlim(-2.5,2.5,0.5)
plt.ylim(-4.5,4.5,0.5)
plt.vlines(0, -4.5, 4.5, lw=1, color='g')
plt.hlines(0, -4.5, 4.5, lw=1, color='g')

plt.fill_between(xmin,ymin,ymax,alpha=0.3,color='tab:orange')
#plt.title('d) SATI vs ENSO', loc = 'left')
# plt.title('a) SATI vs ENSO', loc = 'left')
plt.title('b) SATI vs ENSO', loc = 'left', fontsize=14)
plt.title('SEAS5', loc = 'right')
plt.xlabel('Niño3.4(°C)')
plt.ylabel('Temperature(°C)')
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
plt.text(1.1, 4, 'SEAS5-mem:', style ='italic',
         fontsize = 9.5, color ="black", weight = 'bold')
plt.text(1.1, 3.6, 'CC= %s'%label2, style ='italic',
         fontsize = 9.5, color ="black")
plt.text(1.1, 3.2, 'Slope= %s'%label3, style ='italic',
         fontsize = 9.5, color ="black")

plt.text(1.1, 2.5, 'SEAS5-Ensmean:', style ='italic',
         fontsize = 9.5, color ="black", weight = 'bold')
plt.text(1.1, 2.1, 'CC= %s'%label4, style ='italic',
         fontsize = 9.5, color ="black")
plt.text(1.1, 1.6, 'Slope= %s'%label5, style ='italic',
         fontsize = 9.5, color ="black")
#---------------------Regression SATI vs SST ERA5----------------------------
import numpy as np
from netCDF4 import Dataset
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as m
import mpl_toolkits.basemap as bm
f = Dataset('data path and file')
g = Dataset('data path and file')

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

#here we are defining the imjex

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
lonvar2 = g.variables['lon'][:]
latvar2 = g.variables['lat'][:]
tas2 = g.variables['sst'][:,:,:]

msst2=np.mean(tas2,axis=0)
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
t90 = stats.t.ppf(1-0.05, df-2)
t95 = stats.t.ppf(1-0.025,df-2)

print(t90)
print(t95)
############# t test emj here ####

lat1=np.min(latvar2)
lat2=np.max(latvar2)
lon1=np.min(lonvar2)
lon2=np.max(lonvar2)

[lonall, latall] = np.meshgrid(lonvar2[:], latvar2[:])
ax3 = fig.add_subplot(2, 2, 3)
m1 = bm.Basemap(projection='cyl',llcrnrlat=-40, llcrnrlon=0,urcrnrlat=40, urcrnrlon=361, lon_0=-80, lat_0=0, resolution='l')

#m.drawcoastlines()
m1.drawcoastlines(linewidth=0.25)		# Draw coastal boumjaries
m1.fillcontinents(color='gray',lake_color='white',zorder=2)

parallels = np.arange(-40,41,20.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-180,180,60.) # make longitude lines every 5 degrees from 95W to 70W
m1.drawparallels(parallels,labels=[1,0,0,0],fontsize=12,linewidth=0.01)
m1.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12,linewidth=0.01)


x, y = m(lonall, latall)

levels=[-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6 ]
jjj=m1.contourf(x,y,slopes[:,:], levels, extemj = 'both', cmap=plt.cm.RdBu_r)

plt.contourf(x,y,tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extemj='both',
        colors = 'none', hatches=['...',None,None, None, '...'],alpha=0)
plt.title("c) Reg.(SATI,SST)", loc='left', fontsize=14)
#plt.title("Reg.(SATI,SST)", loc='left', fontsize=10)
plt.title("ERA5", loc='right', fontsize=14)
#---------------Regression SATI vs SST SEAS%

dir1= 'data path'
dir2= 'data path'
bslope = np.zeros((25,181,360))
bslope1 = np.zeros((25,181,360))
bcorrs = np.zeros((25,181,360))


count=0
tmn = 0.0
for i in range(25):
#    print(i)
    i+=1
    print(i)
    m2 = Dataset(dir1+'sys5_e'+str(i)+'.nc')
    d1 = Dataset(dir2+'sys5_e'+str(i)+'.nc')
    time = m2.variables['time']
    lon = m2.variables['lon'][:]
    lat = m2.variables['lat'][:]
    may = m2.variables['m1'][:,:,:]
    jun = m2.variables['m2'][:,:,:]
    mj= (may+jun)/2
    
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
#    print(msst.shape)
    anom=[]
    anom = mj1-msst
    
    lonx, latx = np.meshgrid(lonT, latT)
    weights = np.cos(latx * np.pi / 180.)

    tas_avg = np.zeros(nt)

    for it in np.arange(nt):
        tas_avg[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    #    tas_avg[it] = np.ma.average(enanom[it, -5:5,190:240], weights=weights[-5:5,190:240])
    tas_avg1 = tas_avg
    tas_avg2 = tas_avg /np.std(tas_avg)
    # print("--------------")
    # print(tas_avg1)
    # print("=============")
    tmn = tmn + tas_avg1
    # print(tmn)
    panom=[]
    panom =(may+jun)/2-np.mean((may+jun)/2,axis=0)
    for x in range(0, panom.shape[1]):
        for y in range(0, panom.shape[2]):
            punto = panom[:,x,y]
            slope, intercept, r_value, p_value, std_err = stats.linregress(tas_avg2,punto)
            bslope[count,x,y] = slope
            slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(tas_avg1,punto)
            bcorrs[count,x,y] = r_value
                      
    count+=1

#print(bslope.shape)
bsst=np.mean(bslope,axis=0)
#bsst1=np.mean(bslope1,axis=0)
acorr1=np.mean(bcorrs,axis=0)
            
    #--------------------------------
ax3 = fig.add_subplot(2, 2, 4)
#domain definition plotting
lat1=-40
lat2=41
lon1=0
lon2=361
[lonall, latall] = np.meshgrid(lon[:], lat[:])
mapproj = bm.Basemap(projection='cyl',llcrnrlat=lat1, llcrnrlon=lon1,urcrnrlat=lat2, urcrnrlon=lon2, lon_0=-180, lat_0=0, resolution='l')

mapproj.drawcoastlines(linewidth=0.25)		# Draw coastal boumjaries
#m.fillcontinents(color='white',lake_color='white',zorder=2)
mapproj.fillcontinents(color='gray',lake_color='white',zorder=2)
x, y = mapproj(lonall, latall)
levels=[-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6]
map2=plt.contourf(x,y,bsst[:,:], levels, cmap=plt.cm.RdBu_r)

####  T test for statistical signifcance test ######
df =1050-2
#print(df)
numt=[]
numt=acorr1[:,:]*np.sqrt(df)
print(numt)
denmt=[]
denmt=np.sqrt(1-pow(acorr1,2))
tscore = []
tscore = (numt/denmt)
t90 = stats.t.ppf(1-0.05, df)
t95 = stats.t.ppf(1-0.025,df)

print(t90)
print(t95)
parallels = np.arange(-40,41,20.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-180,180,60.) # make longitude lines every 5 degrees from 95W to 70W
mapproj.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.01)
mapproj.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.01)
plt.contourf(x,y,tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extemj='both',
        colors = 'none', hatches=['...',None,None, None, '...'],alpha=0)
plt.title("d) Reg.(SATI,SST)", loc='left', fontsize=14)
plt.title("SEAS5", loc='right', fontsize=14)
cbar_ax = fig.add_axes([0.2, 0.07, 0.6, 0.04])  # [left, bottom, width, height]
cbar = plt.colorbar(jjj, cax=cbar_ax, orientation='horizontal', shrink=1)
cbar.ax.tick_params(labelsize=12) 
fig.tight_layout()
plt.savefig('figure4abcd.png', dpi = 300)
#plt.tight_layout()
