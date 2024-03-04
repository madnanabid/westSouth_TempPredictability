# -*- coding: utf-8 -*-
"""
Created on Fri jun 23 09:11:04 2023

@author:
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

#diri_output="./"
fig = plt.figure(figsize=(6,5))

diri="Data Path"
diri2="Data Path"
diri3="Data Path"
diri4="Data Path"
diri5="Data Path"

diri6="Data Path"
diri7="Data Path"
#diri8 = "E:/DATA/statn-data-Tmax/mj/tmean/"


obs_era =nc.Dataset(diri2+"filename","r")
obs_cru =nc.Dataset(diri3+"filename","r")
obs_merra2 =nc.Dataset(diri4+"filename","r")
obs_ncep =nc.Dataset(diri5+"filename","r")
obs_cpc =nc.Dataset(diri6+"filename","r")
obs_UoD = nc.Dataset(diri7+"filename","r")
#obs_PAK = nc.Dataset(diri8+"dt.pak.bin.nc","r")
#obs_PAK = nc.Dataset(diri8+"dt.tmn8122.bin.nc","r")

tmp_e1 = nc.Dataset(diri+"sys5_e1.nc","r")
tmp_e2 = nc.Dataset(diri+"sys5_e2.nc","r")
tmp_e3 = nc.Dataset(diri+"sys5_e3.nc","r")
tmp_e4 = nc.Dataset(diri+"sys5_e4.nc","r")
tmp_e5 = nc.Dataset(diri+"sys5_e5.nc","r")
tmp_e6 = nc.Dataset(diri+"sys5_e6.nc","r")
tmp_e7 = nc.Dataset(diri+"sys5_e7.nc","r")
tmp_e8 = nc.Dataset(diri+"sys5_e8.nc","r")
tmp_e9 = nc.Dataset(diri+"sys5_e9.nc","r")
tmp_e10 = nc.Dataset(diri+"sys5_e10.nc","r")
tmp_e11 = nc.Dataset(diri+"sys5_e11.nc","r")
tmp_e12 = nc.Dataset(diri+"sys5_e12.nc","r")
tmp_e13 = nc.Dataset(diri+"sys5_e13.nc","r")
tmp_e14 = nc.Dataset(diri+"sys5_e14.nc","r")
tmp_e15 = nc.Dataset(diri+"sys5_e15.nc","r")
tmp_e16 = nc.Dataset(diri+"sys5_e16.nc","r")
tmp_e17 = nc.Dataset(diri+"sys5_e17.nc","r")
tmp_e18 = nc.Dataset(diri+"sys5_e18.nc","r")
tmp_e19 = nc.Dataset(diri+"sys5_e19.nc","r")
tmp_e20 = nc.Dataset(diri+"sys5_e20.nc","r")
tmp_e21 = nc.Dataset(diri+"sys5_e21.nc","r")
tmp_e22 = nc.Dataset(diri+"sys5_e22.nc","r")
tmp_e23 = nc.Dataset(diri+"sys5_e23.nc","r")
tmp_e24 = nc.Dataset(diri+"sys5_e24.nc","r")
tmp_e25 = nc.Dataset(diri+"sys5_e25.nc","r")

#print(tmp_file.variables)
#may   = tmp_file.variables['m2']
#jun    = tmp_file.variables['m2']
lat     = obs_era.variables['lat'][:]
lon     = obs_era.variables['lon'][:]
time    = obs_era.variables['time'][:]


era=obs_era.variables['t2m'][:,:,:]
era = era -273.15
cru=obs_cru.variables['tmp'][:,:,:]

merra2=obs_merra2.variables['t2m'][:,:,:]
merra2 = merra2 - 273.15
ncep=obs_ncep.variables['air'][:,:,:]
cpc=obs_cpc.variables['tmax'][:,:,:]
UoD=obs_UoD.variables['air'][:,:,:]
#tmpak=obs_PAK.variables['tmp'][:,0,0]

maye1 = tmp_e1.variables['m1'][:,:,:]
june1 = tmp_e1.variables['m2'][:,:,:]

msste1=np.mean((maye1+june1)/2,axis=0)
#print(msst.shape)
anom1=[]
anom1=(maye1+june1)/2-msste1

#print(anom2)
maye2 = tmp_e2.variables['m1'][:,:,:]
june2 = tmp_e2.variables['m2'][:,:,:]

msste2=np.mean((maye2+june2)/2,axis=0)
#print(msst.shape)
anom2=[]
anom2=(maye2+june2)/2-msste2

maye3 = tmp_e3.variables['m1'][:,:,:]
june3 = tmp_e3.variables['m2'][:,:,:]

msste3=np.mean((maye3+june3)/2,axis=0)
#print(msst.shape)
anom3=[]
anom3=(maye3+june3)/2-msste3


maye4 = tmp_e4.variables['m1'][:,:,:]
june4 = tmp_e4.variables['m2'][:,:,:]

msste4=np.mean((maye4+june4)/2,axis=0)
#print(msst.shape)
anom4=[]
anom4=(maye4+june4)/2-msste4


maye5 = tmp_e5.variables['m1'][:,:,:]
june5 = tmp_e5.variables['m2'][:,:,:]

msste5=np.mean((maye5+june5)/2,axis=0)
#print(msst.shape)
anom5=[]
anom5=(maye5+june5)/2-msste5


maye6 = tmp_e6.variables['m1'][:,:,:]
june6 = tmp_e6.variables['m2'][:,:,:]

msste6=np.mean((maye6+june6)/2,axis=0)
#print(msst.shape)
anom6=[]
anom6=(maye6+june6)/2-msste6

maye7 = tmp_e7.variables['m1'][:,:,:]
june7 = tmp_e7.variables['m2'][:,:,:]

msste7=np.mean((maye7+june7)/2,axis=0)
#print(msst.shape)
anom7=[]
anom7=(maye7+june7)/2-msste7

maye8 = tmp_e8.variables['m1'][:,:,:]
june8 = tmp_e8.variables['m2'][:,:,:]

msste8=np.mean((maye8+june8)/2,axis=0)
#print(msst.shape)
anom8=[]
anom8=(maye8+june8)/2-msste8


maye9 = tmp_e9.variables['m1'][:,:,:]
june9 = tmp_e9.variables['m2'][:,:,:]

msste9=np.mean((maye9+june9)/2,axis=0)
#print(msst.shape)
anom9=[]
anom9=(maye9+june9)/2-msste9

maye10 = tmp_e10.variables['m1'][:,:,:]
june10 = tmp_e10.variables['m2'][:,:,:]

msste10=np.mean((maye10+june10)/2,axis=0)
#print(msst.shape)
anom10=[]
anom10=(maye10+june10)/2-msste10


maye11 = tmp_e11.variables['m1'][:,:,:]
june11 = tmp_e11.variables['m2'][:,:,:]

msste11=np.mean((maye11+june11)/2,axis=0)
#print(msst.shape)
anom11=[]
anom11=(maye11+june11)/2-msste11
maye12 = tmp_e12.variables['m1'][:,:,:]
june12 = tmp_e12.variables['m2'][:,:,:]

msste12=np.mean((maye12+june12)/2,axis=0)
#print(msst.shape)
anom12=[]
anom12=(maye12+june12)/2-msste12

maye13 = tmp_e13.variables['m1'][:,:,:]
june13 = tmp_e13.variables['m2'][:,:,:]

msste13=np.mean((maye13+june13)/2,axis=0)
#print(msst.shape)
anom13=[]
anom13=(maye13+june13)/2-msste13

maye14 = tmp_e14.variables['m1'][:,:,:]
june14 = tmp_e14.variables['m2'][:,:,:]

msste14=np.mean((maye14+june14)/2,axis=0)
#print(msst.shape)
anom14=[]
anom14=(maye14+june14)/2-msste14

maye15 = tmp_e15.variables['m1'][:,:,:]
june15 = tmp_e15.variables['m2'][:,:,:]

msste15=np.mean((maye15+june15)/2,axis=0)
#print(msst.shape)
anom15=[]
anom15=(maye15+june15)/2-msste15


maye16 = tmp_e16.variables['m1'][:,:,:]
june16 = tmp_e16.variables['m2'][:,:,:]

msste16=np.mean((maye16+june16)/2,axis=0)
#print(msst.shape)
anom16=[]
anom16=(maye16+june16)/2-msste16


maye17 = tmp_e17.variables['m1'][:,:,:]
june17 = tmp_e17.variables['m2'][:,:,:]

msste17=np.mean((maye17+june17)/2,axis=0)
#print(msst.shape)
anom17=[]
anom17=(maye17+june17)/2-msste17

maye18 = tmp_e18.variables['m1'][:,:,:]
june18 = tmp_e18.variables['m2'][:,:,:]

msste18=np.mean((maye18+june18)/2,axis=0)
#print(msst.shape)
anom18=[]
anom18=(maye18+june18)/2-msste18


maye19 = tmp_e19.variables['m1'][:,:,:]
june19 = tmp_e19.variables['m2'][:,:,:]

msste19=np.mean((maye19+june19)/2,axis=0)
#print(msst.shape)
anom19=[]
anom19=(maye19+june19)/2-msste19


maye20 = tmp_e20.variables['m1'][:,:,:]
june20 = tmp_e20.variables['m2'][:,:,:]

msste20=np.mean((maye20+june20)/2,axis=0)
#print(msst.shape)
anom20=[]
anom20=(maye20+june20)/2-msste20


maye21 = tmp_e21.variables['m1'][:,:,:]
june21 = tmp_e21.variables['m2'][:,:,:]

msste21=np.mean((maye21+june21)/2,axis=0)
#print(msst.shape)
anom21=[]
anom21=(maye21+june21)/2-msste21

maye22 = tmp_e22.variables['m1'][:,:,:]
june22 = tmp_e22.variables['m2'][:,:,:]

msste22=np.mean((maye22+june22)/2,axis=0)
#print(msst.shape)
anom22=[]
anom22=(maye22+june22)/2-msste22

maye23 = tmp_e23.variables['m1'][:,:,:]
june23 = tmp_e23.variables['m2'][:,:,:]

msste23=np.mean((maye23+june23)/2,axis=0)
#print(msst.shape)
anom23=[]
anom23=(maye23+june23)/2-msste23
maye24 = tmp_e24.variables['m1'][:,:,:]
june24 = tmp_e24.variables['m2'][:,:,:]

msste24=np.mean((maye24+june24)/2,axis=0)
#print(msst.shape)
anom24=[]
anom24=(maye24+june24)/2-msste24

maye25 = tmp_e25.variables['m1'][:,:,:]
june25 = tmp_e25.variables['m2'][:,:,:]

msste25=np.mean((maye25+june25)/2,axis=0)
#print(msst.shape)
anom25=[]
anom25=(maye25+june25)/2-msste2

modensmean = (msste1+msste2+msste3+msste4+msste5+msste6+msste7+msste8+msste9+msste10+msste11+msste12+msste13+msste14+msste15+msste16+msste17+msste18+msste19+msste20+msste21+msste22+msste23+msste24+msste25)/25
modensmean = modensmean -273.15
eramean=np.mean(era,axis=0)
crumean=np.mean(cru,axis=0)
merra2mean=np.mean(merra2,axis=0)
ncepmean=np.mean(ncep,axis=0)
cpcmean=np.mean(cpc,axis=0)
UoDmean=np.mean(UoD,axis=0)
#tmpakmean=np.mean(tmpak,axis=0)
nt, nlat, nlon = era.shape

nt2, nlat2, nlon2 = cru.shape
nt3, nlat3, nlon3 = UoD.shape

ngrd = nlon*nlat
# print(nt)
# print(ngrd)
era_bias =  modensmean - eramean
cpc_bias =  modensmean - cpcmean 
ncep_bias = modensmean -  ncepmean 
cru_bias =  modensmean - crumean
merra2_bias =  modensmean - merra2mean

lonE=65
lonW=80
latS=31
latN=39

# lonE=70
# lonW=75
# latS=31
# latN=36
lon1=np.where(lon==lonE)
lon2=np.where(lon==lonW)
lat1=np.where(lat==latS)
lat2=np.where(lat==latN)
#print(lon1,lon2,lat1,lat2)
lonx, latx = np.meshgrid(lon, lat)
weights = np.cos(latx * np.pi / 180.)
   
#ts_sati=[]
#for it2 in np.arange(nt2):
ts_era = np.ma.average(era_bias[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
ts_cru = np.ma.average(cru_bias[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])

ts_ncep = np.ma.average(ncep_bias[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
ts_merra2 = np.ma.average(merra2_bias[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
ts_cpc = np.ma.average(cpc_bias[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])

print(ts_cru)
print(ts_ncep)
print(ts_merra2)
print(ts_cpc)



# Values for the bar plot
labels = ['ERA', 'CRU', 'NCEP', 'MERRA2', 'CPC']
values = [ts_era, ts_cru, ts_ncep, ts_merra2, ts_cpc]
bar_width = 0.4  # Adjust this value to change the bar width
plt.ylim(-4,4)
plt.axhline(0, color='black',ls='--',lw=1)
# Create a bar plot
plt.bar(labels, values, width=bar_width)
plt.xlabel('Datasets',fontsize=12)
plt.ylabel('Bias', fontsize=12)
#plt.title('Bias Comparison')

plt.title('BIAS Comparison (SEAS5) over WSA',fontsize=14,loc = "left")
plt.savefig('BIAS_Comparison_mod.png', dpi = 300)
plt.show()
