# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 19:55:57 2023

"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(6,5))
dir1 = "Data Path"
dir2 = "Data Path"
dir3 = "Data Path"


f = Dataset(dir1+'filename')
g = Dataset(dir2+'filename')
h = Dataset(dir3+'filename')


time = g.variables['time']
lon = g.variables['lon'][:]
lat = g.variables['lat'][:]

may = g.variables['m1'][:,:,:]
jun = g.variables['m2'][:,:,:]
tmp= (may+jun)/2

nt, nlat, nlon = tmp.shape
ngrd = nlon*nlat
# print(nt)
# print(ngrd)
 
lonE=65
lonW=80
latS=31
latN=39

lontmp1=np.where(lon==lonE)
lontmp2=np.where(lon==lonW)
lattmp1=np.where(lat==latS)
lattmp2=np.where(lat==latN)
print(lontmp1,lontmp2,lattmp1,lattmp2)

lonx, latx = np.meshgrid(lon, lat)
weights = np.cos(latx * np.pi / 180.)

ts_sati = np.zeros(nt)

for it in np.arange(nt):
    ts_sati[it] = np.ma.average(tmp[it, int(lattmp1[0]):int(lattmp2[0]),int(lontmp1[0]):int(lontmp2[0])], weights=weights[int(lattmp1[0]):int(lattmp2[0]),int(lontmp1[0]):int(lontmp2[0])])


timesst =f.variables['time']
lonsst = f.variables['lon'][:]
latsst = f.variables['lat'][:]
sst = h.variables['sst'][:,:,:]

# uncomment for model SSTs
may1 = f.variables['m1'][:,:,:]
jun1 = f.variables['m2'][:,:,:]
sst1= (may1+jun1)/2
nt, nlat, nlon = sst1.shape
ngrd = nlon*nlat
# print(nt)
# print(ngrd)

lonE=190
lonW=240
latS=-5
latN=5

lon1=np.where(lonsst==lonE)
lon2=np.where(lonsst==lonW)
lat1=np.where(latsst==latS)
lat2=np.where(latsst==latN)
ts_nino34_obs = np.zeros(nt)
ts_nino34_mod = np.zeros(nt)
for it in np.arange(nt):
    ts_nino34_mod[it] = np.ma.average(sst1[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_nino34_obs[it] = np.ma.average(sst[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])

#here you check your la nina/elnino years values in your timseries
lanina_val_obs= ts_sati[ts_nino34_obs<=0.5*-1.]
elnino_val_obs= ts_sati[ts_nino34_obs>=0.5]

lanina_val_mod= ts_sati[ts_nino34_mod<=0.5*-1.]
elnino_val_mod= ts_sati[ts_nino34_mod>=0.5]

print('OBS')
print(elnino_val_obs.shape,lanina_val_obs.shape)

print('MOD')
print(elnino_val_mod.shape,lanina_val_mod.shape)

print(np.round(elnino_val_mod,2))
print(np.round(elnino_val_mod,2))
#print('##################')
sst_elnino_obs = ts_sati[ts_nino34_obs>=0.5] 
sst_elnino_mod = ts_sati[ts_nino34_mod>=0.5] 
sst_lanina_obs = ts_sati[ts_nino34_obs<=0.5*-1.]
sst_lanina_mod = ts_sati[ts_nino34_mod<=0.5*-1.]
#print(sst_lanina.shape)
fig, ax = plt.subplots(figsize=(8,5))

ax1 = sns.distplot(ts_sati,
                  kde=True,
                  hist = False,
                  color='black',
                  label="Climo")


#ax1 = sns.displot(mj1d,x="temperature (C)",kimj="kde")
ax1 = sns.distplot(sst_elnino_obs,
                  kde=True,
                  hist = False,
                  color='blue', label="El Niño (ERA5)")
ax1 = sns.distplot(sst_lanina_obs,
                  kde=True,
                  hist = False,
                  color='red', label="La Niña (ERA5)")


ax1 = sns.distplot(sst_elnino_mod,
                  kde=True,
                  hist = False,
                  color='blue', kde_kws={'linestyle':'--'}, label="El Niño (SEAS5)")
ax1 = sns.distplot(sst_lanina_mod,
                  kde=True,
                  hist = False,
                  color='red', kde_kws={'linestyle':'--'}, label="La Niña (SEAS5)")

ax1.legend(prop={'size': 12}, fontsize='medium',loc='upper left',ncol=1)

ax1.vlines(0,0,1, lw=1, ls='--', color='black')
plt.xlabel('SAT (°C)', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.ylim(0,0.6)
#plt.xlim(-3,3)
plt.xlim(-3.5,3.5,0.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
xmin=[-0.5,0.5]
ymin=[0]
ymax=[1]

plt.fill_between(xmin,ymin,ymax,alpha=0.3,color='tab:grey')
plt.title('PDF (SAT vs ENSO)',loc='left', fontsize=16)
plt.savefig('PDF_plot_total_elnino_Lanina_both-OBS-MOD.png',dpi=300)
plt.show()

