# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 18:26:48 2023

@author:
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from netCDF4 import Dataset

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
    lat = m2.variables['lat'][:]
    may = m2.variables['m1'][:,:,:]
    dec = m2.variables['m2'][:,:,:]
    nd= (may+dec)/2


    lonT = d1.variables['lon'][:]
    latT = d1.variables['lat'][:]
    may1 = d1.variables['m1'][:,:,:]
    dec1 = d1.variables['m2'][:,:,:]
    mj1= (may1+dec1)/2
    
    
    nt, nlat, nlon = mj1.shape
    ngrd = nlon*nlat
    print(mj1.shape)


    lonE=190 ; lonW=240; latS=-5; latN=5
    lon1=np.where(lonT==lonE)
    lon2=np.where(lonT==lonW)
    lat1=np.where(latT==latS)
    lat2=np.where(latT==latN)
    msst=np.mean(mj1,axis=0)
    print(lon1,lon2,lat1,lat2)
#    print()
#    print(msst.shape)

    anom=[]
    anom = mj1-msst
    
    lonx, latx = np.meshgrid(lonT, latT)
    weights = np.cos(latx * np.pi / 180.)
    tas_avg = np.zeros(nt)

    for it in np.arange(nt):
       tas_avg[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    tas_avg = tas_avg /np.std(tas_avg)
    # bnino34 = anom[:,int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])]
    # tsnino34=np.mean(bnino34,axis=(1,2))
    # #standardized anomaly index

    # nindex34=tsnino34/np.std(tsnino34)
#    quit()
#quit()
    panom=[]
    panom =(may+dec)/2-np.mean((may+dec)/2,axis=0)
#    panom =nd1-np.mean((1)/2,axis=0)
#    print(panom)
#    panom = np.ma.anomalies(m1.variables['m1'][:,:,:])
    for x in range(0, panom.shape[1]):
        for y in range(0, panom.shape[2]):
            punto = panom[:,x,y]
            slope, intercept, r_value, p_value, std_err = stats.linregress(tas_avg,punto)
            bslope[count,x,y] = slope
            #bcorrs[count,x,y] = (0.5*(log(1.+r_value)-log(1.-r_value)))
    count+=1

lonEW=65 ; lonWW=80; latSW=31; latNW=39
lon3=np.where(lonT==lonEW)
lon4=np.where(lonT==lonWW)
lat3=np.where(latT==latSW)
lat4=np.where(latT==latNW)
#msst=np.mean(nd1,axis=0)
print(lon3,lon4,lat3,lat4)
val = []
data = []
for i in range (25):
    val = np.ma.average(bslope[i, int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
    print(np.round(val,2))
    data.append(val * -1)
    #data = val * -1
#data = data * -1
print(data)
# x=['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7','e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14','e15', 'e16', 'e17', 'e18', 'e19', 'e20', 'e21','e22', 'e23', 'e24', 'e25']
# barlist = plt.bar(x, data , width = 0.5, label='Ensemble Members x (-1)', color='blue')

fig, ax = plt.subplots(figsize=(8, 5))  # Adjust the width (12) and height (6) as per your preferences

std = np.std(data)
#std2 = std*2
print(std)

#udata=data+std
#ddata=data-std
ensmean = np.mean(data)
print(np.round(ensmean,2))
above = ensmean + std
below = ensmean - std
print(above)
print(below)
#print(above)
#print(below)
plt.ylim(0,0.6,0.2)
x=['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7','e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14','e15', 'e16', 'e17', 'e18', 'e19', 'e20', 'e21','e22', 'e23', 'e24', 'e25']
barlist = plt.bar(x, data, width = 0.5, label='Ensemble Members x (-1)', color='blue')
plt.hlines(0.44, 0, 24, label = 'ERA5 x (-1)' ,color='black',linewidth=1)
#plt.hlines(0.45, 0, 26, label = 'ERA5' ,color='black',linewidth=1)
plt.hlines(above, 0, 24, label='EnsMean \u00B1 STD', color='red', linestyle='--', linewidth=1)
# plt.hlines(above, 0, 26, label = 'EnsMean  STD', color='red', linestyle='--',linewidth=1)
plt.hlines(below, 0, 24, color='red', linestyle='--',linewidth=1)
# Add the legend with the updated labels
#plt.legend(prop={'size': 9}, fontsize='medium', loc='upper left', ncol=2)
plt.legend(prop={'size':10}, fontsize='large',loc='upper left',ncol=2)
plt.xticks(fontsize=12, rotation = 90)  # Adjust fontsize as needed for x-ticks
plt.yticks(fontsize=12)  # Adjust fontsize as needed for y-ticks
#plt.fill_between(4, 10, 30, alpha=0.2)
plt.title("SAT regression coefficient over WSA", loc='left', fontsize=14)
plt.savefig('std_reg_coef_tmp_Rev.png', dpi = 300)

#quit()
